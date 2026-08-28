import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import BadRequestError
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..llm.openai_gpt import CacheOpenAI, _safe_error_summary

logger = get_logger(__name__)


class ResponseFormatError(ValueError):
    """LLM 响应中未找到预期的 JSON 结构。"""
    pass


_RESP_SNIPPET_LEN = 500


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_ner_from_response(real_response):
    # Try strict pattern first (no nested braces)
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, real_response, re.DOTALL)
    if match is not None:
        return eval(match.group())["named_entities"]
    # Fallback: find any JSON object with named_entities key (allows nested)
    pattern2 = r'\{.*?"named_entities"\s*:\s*(\[[^\]]*\]).*?\}'
    match2 = re.search(pattern2, real_response, re.DOTALL)
    if match2 is not None:
        try:
            return json.loads(match2.group(1))
        except Exception:
            pass
    # Fallback: bare JSON array
    array_match = re.search(r'\[.*?\]', real_response, re.DOTALL)
    if array_match is not None:
        try:
            result = json.loads(array_match.group())
            if isinstance(result, list):
                return result
        except Exception:
            pass
    snippet = (real_response or "")[:_RESP_SNIPPET_LEN]
    raise ResponseFormatError(f"响应中未找到 named_entities JSON 结构；响应片段: {snippet!r}")


class OpenIE:
    # ThreadPoolExecutor 默认并发数是 min(32, cpu_count+4)，在核数多的机器上会
    # 对 LLM 代理（尤其是第三方聚合 API，比如 302ai）打出很高的并发，容易触发
    # 限流/超载，表现为大量 400 报错甚至代理返回损坏的响应体。这里给一个更
    # 保守的默认并发上限，需要更高吞吐可以在构造 OpenIE 时传 max_workers 覆盖。
    DEFAULT_MAX_WORKERS =10

    def __init__(self, llm_model: CacheOpenAI, max_workers: int = DEFAULT_MAX_WORKERS):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        self.max_workers = max_workers
        self.last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "cache_hits": 0,
        }

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=ner_input_message,
            )
            metadata['cache_hit'] = cache_hit
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_entities = _extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))

        except Exception as e:
            summary = _safe_error_summary(e)
            if isinstance(e, BadRequestError):
                logger.debug(f"NER chunk {chunk_key} 因 BadRequest 失败（已在 infer 记录）: {summary}")
            else:
                logger.warning(f"NER extraction failed for chunk {chunk_key}: {summary}")
            metadata.update({'error': summary})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                unique_entities=[],
                metadata=metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:
        def _extract_triples_from_response(real_response):
            pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
            match = re.search(pattern, real_response, re.DOTALL)
            if match is None:
                snippet = (real_response or "")[:_RESP_SNIPPET_LEN]
                raise ResponseFormatError(f"响应中未找到 triples JSON 结构；响应片段: {snippet!r}")
            return eval(match.group())["triples"]

        # PREPROCESSING
        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
            )
            metadata['cache_hit'] = cache_hit
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_triples = _extract_triples_from_response(real_response)
            triplets = filter_invalid_triples(triples=extracted_triples)

        except Exception as e:
            summary = _safe_error_summary(e)
            if isinstance(e, BadRequestError):
                logger.debug(f"Triple extraction chunk {chunk_key} 因 BadRequest 失败（已在 infer 记录）: {summary}")
            else:
                logger.warning(f"Triple extraction failed for chunk {chunk_key}: {summary}")
            metadata.update({'error': summary})
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[]
            )

        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk 
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1

                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        ner_usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "calls": len(ner_results_list),
            "cache_hits": num_cache_hit,
        }

        triple_results_list = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(self.triple_extraction, ner_result.chunk_id,
                                chunk_passages[ner_result.chunk_id],
                                ner_result.unique_entities): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        triple_usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "calls": len(triple_results_list),
            "cache_hits": num_cache_hit,
        }
        prompt_tokens = ner_usage["prompt_tokens"] + triple_usage["prompt_tokens"]
        completion_tokens = ner_usage["completion_tokens"] + triple_usage["completion_tokens"]
        self.last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "calls": ner_usage["calls"] + triple_usage["calls"],
            "cache_hits": ner_usage["cache_hits"] + triple_usage["cache_hits"],
            "ner": ner_usage,
            "triple": triple_usage,
        }

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}

        return ner_results_dict, triple_results_dict
