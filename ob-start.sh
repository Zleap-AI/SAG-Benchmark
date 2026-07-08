#!/bin/bash
set -e

INIT_SQL=/root/boot/init.sql
INIT_MARK=/root/ob/.sag2_init_done
SYS_ROOT_PASSWORD=${OB_SYS_ROOT_PASSWORD:-sag2_root}

sys_ready() {
  obclient -h127.0.0.1 -P2881 -uroot@sys -e "SELECT 1" >/dev/null 2>&1 ||
    obclient -h127.0.0.1 -P2881 -uroot@sys -p"$SYS_ROOT_PASSWORD" -e "SELECT 1" >/dev/null 2>&1
}

/usr/sbin/sshd

/root/boot/start.sh &

echo "==> Waiting for OceanBase..."
until sys_ready; do
  sleep 5
done
echo "==> OceanBase ready."

echo "==> Waiting for sag2 tenant..."
until obclient -h127.0.0.1 -P2881 -uroot@sag2 -e "SELECT 1" >/dev/null 2>&1; do
  sleep 5
done
echo "==> sag2 tenant ready."

if [ -f "$INIT_SQL" ] && [ ! -f "$INIT_MARK" ]; then
  obclient -h127.0.0.1 -P2881 -uroot@sag2 < "$INIT_SQL"
  touch "$INIT_MARK"
  echo "==> init.sql executed."
else
  echo "==> init.sql skipped."
fi

echo "==> All done."

wait
