import asyncio
import html
import io
import csv
import json
import logging
import os
import re
import httpx
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union, List
import xml.etree.ElementTree as ET

import numpy as np
import tiktoken

ENCODER = None

logger = logging.getLogger("hyper_rag")


def set_logger(log_file: str, verbose: bool = False):
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # 控制台 handler：默认 WARNING（安静），--verbose 时 DEBUG（展开细粒度日志）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable
    # vLLM /tokenize 端点配置：用 embedding 模型真实分词器做精确截断。
    # 强制要求配置——不配会在调用时报错（避免与模型端 token 计数不一致）。
    tokenize_url: str = None
    tokenize_model: str = None
    tokenize_api_key: str = None

    # 模块级复用的 httpx 客户端（懒加载，所有 EmbeddingFunc 实例共享）
    _http_client: "httpx.AsyncClient" = None

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        # 对输入文本逐条按 token 精确截断到 max_token_size（vLLM /tokenize 计数）
        n = len(args[0]) if (args and isinstance(args[0], list)) else (
            len(kwargs["texts"]) if isinstance(kwargs.get("texts"), list) else 1)
        logger.debug(f"[embedding] 调 embedding 服务：{n} 条文本，先 /tokenize 截断到 ≤{self.max_token_size} token，再向量化")
        if args and isinstance(args[0], list):
            args = (await self._truncate_texts(args[0]),) + args[1:]
        elif "texts" in kwargs and isinstance(kwargs["texts"], list):
            kwargs["texts"] = await self._truncate_texts(kwargs["texts"])
        return await self.func(*args, **kwargs)

    async def _truncate_texts(self, texts: list) -> list:
        limit = self.max_token_size
        if not limit or limit <= 0:
            return texts
        if not self.tokenize_url or not self.tokenize_model:
            raise RuntimeError(
                "EmbeddingFunc 未配置 tokenize_url/tokenize_model。"
                "请在构造 EmbeddingFunc 时传入 vLLM 服务的 tokenize_url 与 tokenize_model"
                "（用于按模型真实分词器精确截断）。"
            )
        # 仅对字符串做截断，非字符串原样保留
        idx_strs = [(i, t) for i, t in enumerate(texts) if isinstance(t, str)]
        if not idx_strs:
            return texts
        # 并发对每条文本调 /tokenize+/detokenize（vLLM /tokenize 只接受单条 string）
        truncated = await asyncio.gather(
            *(self._truncate_one(t, limit) for _, t in idx_strs)
        )
        out = list(texts)
        for (i, _), new_t in zip(idx_strs, truncated):
            out[i] = new_t
        return out

    async def _truncate_one(self, text: str, limit: int) -> str:
        client = await self._get_http_client()
        headers = {}
        if self.tokenize_api_key:
            auth = self.tokenize_api_key
            if not auth.startswith("Bearer "):
                auth = "Bearer " + auth
            headers["Authorization"] = auth
        base = self._base_url()  # 剥掉末尾 /v1（vLLM /tokenize 在根路径）
        # 1) /tokenize 拿真实 token id
        tok_resp = await self._post_with_retry(
            client,
            f"{base}/tokenize",
            {"model": self.tokenize_model, "prompt": text},
            headers,
        )
        if tok_resp.status_code != 200:
            raise RuntimeError(
                f"vLLM /tokenize 失败 (HTTP {tok_resp.status_code}): {tok_resp.text[:200]}"
            )
        tokens = tok_resp.json().get("tokens", [])
        if len(tokens) <= limit:
            return text  # 未超限，原样返回
        # 2) /detokenize 把前 limit 个 token id 转回文本
        det_resp = await self._post_with_retry(
            client,
            f"{base}/detokenize",
            {"model": self.tokenize_model, "tokens": tokens[:limit]},
            headers,
        )
        if det_resp.status_code != 200:
            raise RuntimeError(
                f"vLLM /detokenize 失败 (HTTP {det_resp.status_code}): {det_resp.text[:200]}"
            )
        return det_resp.json().get("prompt", text)

    async def _post_with_retry(self, client, url, payload, headers, retries: int = 3):
        """对 tokenize/detokenize 的 httpx.post 包重试（指数退避）。

        只重试偶发错误：网络抖动（Broken pipe/ReadError/超时）+ 429 + 5xx。
        2xx 或不可重试的 4xx（如 400）原样返回，沿用调用方的状态码判断。
        """
        last_exc = None
        for i in range(retries):
            try:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_exc = RuntimeError(
                        f"transient HTTP {resp.status_code} @ {url}"
                    )
                    await asyncio.sleep(2 ** i)
                    continue
                return resp
            except (
                httpx.ReadError,
                httpx.ConnectError,
                httpx.RemoteProtocolError,
                httpx.PoolTimeout,
                httpx.ReadTimeout,
                httpx.ConnectTimeout,
            ) as e:
                last_exc = e
                logger.warning(f"[tokenize] {url} 第 {i+1}/{retries} 次失败: {type(e).__name__}: {e}，重试")
                await asyncio.sleep(2 ** i)
        raise last_exc if last_exc else RuntimeError(f"{url} 重试 {retries} 次仍失败")

    def _base_url(self) -> str:
        # 兼容 http://host:port 与 http://host:port/v1 两种写法
        u = self.tokenize_url.rstrip("/")
        if u.endswith("/v1"):
            u = u[:-3]
        return u

    @classmethod
    async def _get_http_client(cls):
        if cls._http_client is None:
            import httpx
            # 限制连接池上限 + keepalive 超时回收，避免 CLOSE-WAIT 连接堆积占满池导致卡死。
            # - max_connections：硬上限，防止无限增长
            # - max_keepalive_connections：保活的空闲连接上限
            # - keepalive_expiry：空闲连接 N 秒后自动关闭（回收 CLOSE-WAIT）
            # - pool_timeout：池满时等待空闲连接的超时，避免永久阻塞
            cls._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, pool=10.0),
                limits=httpx.Limits(
                    max_connections=32,
                    max_keepalive_connections=8,
                    keepalive_expiry=30.0,
                ),
                headers={"Connection": "close"},  # 不保活，每次用完即关，杜绝 CLOSE-WAIT 堆积
            )
        return cls._http_client



def locate_json_string_body_from_string(content: str) -> Union[str, None]:
    """Locate the JSON string body from a string"""
    maybe_json_str = re.search(r"{.*}", content, re.DOTALL)
    if maybe_json_str is not None:
        return maybe_json_str.group(0)
    else:
        return None


def convert_response_to_json(response: str) -> dict:
    json_str = locate_json_string_body_from_string(response)
    assert json_str is not None, f"Unable to parse JSON from response: {response}"
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}")
        raise e from None


def compute_args_hash(*args):
    return md5(str(args).encode()).hexdigest()


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001, timeout: float = None):
    """Add restriction of maximum async calling times for a async func.

    timeout: 若设了，对「真正执行 func」的阶段加 asyncio.wait_for 超时。
    关键：超时只计拿到名额后的执行时间，**排队等待（while 循环）不计入**。
    这样大批量并发时，排队久的任务不会被误判超时。
    """

    def final_decro(func):
        """Not using async.Semaphore to aovid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            try:
                if timeout is not None:
                    # 超时只罩「执行 func」阶段（已拿到并发名额），排队时间不计。
                    # 对 asyncio.TimeoutError 额外重试 timeout_retries 次（卡死 rescue），
                    # 退避 4→8→... 秒；重试时仍占着这个并发名额（接受短期吞吐下降，
                    # 换取卡死 chunk 的 rescue 机会）。重试耗尽才抛给容错层跳过。
                    timeout_retries = 2
                    last_timeout = None
                    for attempt in range(timeout_retries + 1):
                        try:
                            result = await asyncio.wait_for(func(*args, **kwargs), timeout=timeout)
                            return result
                        except asyncio.TimeoutError as e:
                            last_timeout = e
                            if attempt < timeout_retries:
                                backoff = 4 * (2 ** attempt)  # 4s, 8s
                                logger.warning(
                                    f"[llm-timeout-retry] 调用超时(>{timeout}s)，第 {attempt+1}/{timeout_retries} 次重试，退避 {backoff}s（仍占并发名额）"
                                )
                                await asyncio.sleep(backoff)
                            else:
                                logger.warning(
                                    f"[llm-timeout-retry] 调用超时重试 {timeout_retries} 次仍失败，抛给上层跳过"
                                )
                    raise last_timeout
                else:
                    result = await func(*args, **kwargs)
                    return result
            finally:
                __current_size -= 1

        return wait_func

    return final_decro

def limit_async_gen_call(max_size: int):
    """
    限制“异步生成器（async generator）”并发数的装饰器。
    适用于 stream 场景：func(*args, **kwargs) 返回一个 async generator，
    不能对其 await，只能 async for 迭代。
    """
    sem = asyncio.Semaphore(max_size)

    def final_decro(func):
        @wraps(func)
        async def gen_wrapper(*args, **kwargs):
            await sem.acquire()
            try:
                agen = func(*args, **kwargs)  # 注意：这里不要 await
                async for item in agen:
                    yield item
            finally:
                sem.release()

        return gen_wrapper

    return final_decro


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


def load_json(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)


def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    content = ENCODER.decode(tokens)
    return content


def pack_user_ass_to_openai_messages(*args: str):
    roles = ["user", "assistant"]
    return [
        {"role": roles[i % 2], "content": content} for i, content in enumerate(args) #if content is not None
    ]


def split_string_by_multi_markers(content: str, markers: list[str]) -> list[str]:
    """Split a string by multiple markers"""
    if not markers:
        return [content]
    results = re.split("|".join(re.escape(marker) for marker in markers), content)
    return [r.strip() for r in results if r.strip()]


# Refer the utils functions of the official GraphRAG implementation:
# https://github.com/microsoft/graphrag
def clean_str(input: Any) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    return re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)


def is_float_regex(value):
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def list_of_list_to_csv(data: List[List[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(data)
    return output.getvalue()


def csv_string_to_list(csv_string: str) -> List[List[str]]:
    output = io.StringIO(csv_string)
    reader = csv.reader(output)
    return [row for row in reader]


def save_data_to_file(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def xml_to_json(xml_file):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Print the root element's tag and attributes to confirm the file has been correctly loaded
        print(f"Root element: {root.tag}")
        print(f"Root attributes: {root.attrib}")

        data = {"nodes": [], "edges": []}

        # Use namespace
        namespace = {"": "http://graphml.graphdrawing.org/xmlns"}

        for node in root.findall(".//node", namespace):
            node_data = {
                "id": node.get("id").strip('"'),
                "entity_type": node.find("./data[@key='d0']", namespace).text.strip('"')
                if node.find("./data[@key='d0']", namespace) is not None
                else "",
                "description": node.find("./data[@key='d1']", namespace).text
                if node.find("./data[@key='d1']", namespace) is not None
                else "",
                "source_id": node.find("./data[@key='d2']", namespace).text
                if node.find("./data[@key='d2']", namespace) is not None
                else "",
            }
            data["nodes"].append(node_data)

        for edge in root.findall(".//edge", namespace):
            edge_data = {
                "source": edge.get("source").strip('"'),
                "target": edge.get("target").strip('"'),
                "weight": float(edge.find("./data[@key='d3']", namespace).text)
                if edge.find("./data[@key='d3']", namespace) is not None
                else 0.0,
                "description": edge.find("./data[@key='d4']", namespace).text
                if edge.find("./data[@key='d4']", namespace) is not None
                else "",
                "keywords": edge.find("./data[@key='d5']", namespace).text
                if edge.find("./data[@key='d5']", namespace) is not None
                else "",
                "source_id": edge.find("./data[@key='d6']", namespace).text
                if edge.find("./data[@key='d6']", namespace) is not None
                else "",
            }
            data["edges"].append(edge_data)

        # Print the number of nodes and edges found
        print(f"Found {len(data['nodes'])} nodes and {len(data['edges'])} edges")

        return data
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def process_combine_contexts(hl, ll):
    header = None
    list_hl = csv_string_to_list(hl.strip())
    list_ll = csv_string_to_list(ll.strip())

    if list_hl:
        header = list_hl[0]
        list_hl = list_hl[1:]
    if list_ll:
        header = list_ll[0]
        list_ll = list_ll[1:]
    if header is None:
        return ""

    if list_hl:
        list_hl = [",".join(item[1:]) for item in list_hl if item]
    if list_ll:
        list_ll = [",".join(item[1:]) for item in list_ll if item]

    combined_sources_set = set(filter(None, list_hl + list_ll))

    combined_sources = [",\t".join(header)]

    for i, item in enumerate(combined_sources_set, start=1):
        combined_sources.append(f"{i},\t{item}")

    combined_sources = "\n".join(combined_sources)

    return combined_sources


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure that there is always an event loop available.

    This function tries to get the current event loop. If the current event loop is closed or does not exist,
    it creates a new event loop and sets it as the current event loop.

    Returns:
        asyncio.AbstractEventLoop: The current or newly created event loop.
    """
    try:
        # Try to get the current event loop
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop

    except RuntimeError:
        # If no event loop exists or it is closed, create a new one
        logger.info("Creating a new event loop in main thread.")
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop

def deduplicate_by_key(data_list, key_string):
    unique_data = []
    seen_keys = set()

    def make_hashable(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            try:
                return tuple(sorted(make_hashable(v) for v in value))
            except TypeError:
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in value.items()))
        return str(value)

    for item in data_list:
        raw_key = item.get(key_string)
        if raw_key is None:
            continue
        key = make_hashable(raw_key)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_data.append(item)
    return unique_data  
