#!/usr/bin/env bash
# Laptop <-> AWS EC2 helper: memory-heavy steps (preprocess, blocking) on the instance, CPU-heavy
# ones (features, training, evaluation) wherever is faster. Large files move through S3 (direct
# scp from the instance was ~50 KB/s); the instance never holds AWS credentials (presigned URLs).
#
#   ./remote.sh start                  start the instance, print its IP
#   ./remote.sh push                   copy the current code (working tree) to the instance
#   ./remote.sh run  "block_test prune_test predict package"   run_pipeline.sh steps, detached
#   ./remote.sh status                 pipeline log + load/memory
#   ./remote.sh pull <path> [...]      remote paths under ~/ber (e.g. output models dataset/processed/candidates_test)
#                                      -> tar -> S3 -> ./remote_pull/ here
#   ./remote.sh stop                   stop (not terminate) the instance: disk and intermediates are kept
#
# Settings (env): BER_INSTANCE (instance id; default: the one tagged Name=claude-ber-cpu),
#                 BER_KEY (~/.ssh/claude-ber.pem), BER_BUCKET (sagemaker-us-east-1-134051031272)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
AWS=${AWS:-$ROOT/venv/bin/aws}
PY=${PY:-$ROOT/venv/bin/python}
KEY=${BER_KEY:-$HOME/.ssh/claude-ber.pem}
BUCKET=${BER_BUCKET:-sagemaker-us-east-1-134051031272}
ID=${BER_INSTANCE:-$($AWS ec2 describe-instances --filters Name=tag:Name,Values=claude-ber-cpu \
     Name=instance-state-name,Values=pending,running,stopping,stopped \
     --query 'Reservations[0].Instances[0].InstanceId' --output text)}
[ "$ID" = None ] && { echo "no instance tagged claude-ber-cpu; set BER_INSTANCE"; exit 1; }
ip() { $AWS ec2 describe-instances --instance-ids "$ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text; }
SSH() { ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "ubuntu@$(ip)" "$@"; }
presign_put() { $PY -c "import boto3,sys; print(boto3.client('s3', region_name='us-east-1').generate_presigned_url('put_object', Params={'Bucket': '$BUCKET', 'Key': sys.argv[1]}, ExpiresIn=6*3600))" "$1"; }

case "${1:-}" in
  start)
    $AWS ec2 start-instances --instance-ids "$ID" >/dev/null
    $AWS ec2 wait instance-running --instance-ids "$ID"
    for i in $(seq 1 30); do SSH true 2>/dev/null && break; sleep 5; done
    echo "running: $ID at $(ip)" ;;
  push)
    tar -czf /tmp/ber_code.tgz -C "$ROOT" --exclude=__pycache__ --exclude=logs code/business_entity_resolution Documentation_template.md
    scp -q -i "$KEY" /tmp/ber_code.tgz "ubuntu@$(ip):~/" && SSH 'mkdir -p ~/ber && tar -xzf ~/ber_code.tgz -C ~/ber && rm ~/ber_code.tgz'
    SSH '[ -x ~/ber/venv/bin/python ] || (cd ~/ber && python3 -m venv venv && venv/bin/pip install -q -r code/business_entity_resolution/requirements.txt)'
    echo "code pushed" ;;
  run)
    shift
    SSH "cd ~/ber/code/business_entity_resolution && mkdir -p logs && MEM_CAP=none PY=../../../venv/bin/python ${REMOTE_ENV:-} setsid nohup ./run_pipeline.sh $* > logs/pipeline.log 2>&1 < /dev/null & sleep 2; cat logs/pipeline.log"
    echo "running detached; ./remote.sh status" ;;
  status)
    SSH 'cat ~/ber/code/business_entity_resolution/logs/pipeline.log; echo; uptime; free -g | sed -n 2p' ;;
  pull)
    shift; mkdir -p "$ROOT/remote_pull"
    k="ber/pull/$(date +%Y%m%d-%H%M%S).tgz"
    url=$(presign_put "$k")
    SSH "cd ~/ber && tar -czf ~/pull.tgz $* && curl -s -o /dev/null -w 'upload http %{http_code}\n' -T ~/pull.tgz '$url' && rm ~/pull.tgz"
    $AWS s3 cp "s3://$BUCKET/$k" /tmp/ber_pull.tgz --only-show-errors && tar -xzf /tmp/ber_pull.tgz -C "$ROOT/remote_pull" && rm /tmp/ber_pull.tgz
    echo "pulled into $ROOT/remote_pull/: $*" ;;
  stop)
    $AWS ec2 stop-instances --instance-ids "$ID" --query 'StoppingInstances[0].CurrentState.Name' --output text ;;
  *) sed -n 2,17p "$0"; exit 1 ;;
esac
