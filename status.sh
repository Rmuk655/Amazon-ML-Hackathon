#!/bin/bash
# One-shot status of everything running for the BER pipeline (AWS instances, jobs on them, Kaggle, submissions).
#   ~/Amazon-ML-Hackathon/status.sh
cd "$(dirname "$0")"
A=venv/bin/aws; K=venv/bin/kaggle; KEY=~/.ssh/claude-ber.pem
echo "=== $(date '+%a %H:%M') ==="
echo "--- AWS instances"
$A ec2 describe-instances --filters Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[].Instances[].[Tags[?Key==`Name`]|[0].Value,InstanceId,State.Name,PublicIpAddress]' --output text
IP=$($A ec2 describe-instances --filters Name=tag:Name,Values=claude-ber-run3 Name=instance-state-name,Values=running \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null)
if [ -n "$IP" ] && [ "$IP" != None ]; then
  timeout 40 ssh -i $KEY -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new ubuntu@$IP '
    L=~/ber/code/business_entity_resolution/logs
    echo "--- run 4 (submission4, no retrain)"; tail -3 $L/run4.log 2>/dev/null
    echo "--- chain (run4 -> analyses -> run 5 preview)"; tail -4 ~/chain.log 2>/dev/null
    echo "--- run 5 preview"; tail -5 $L/run5p.log 2>/dev/null || echo "not started"
    echo "--- run 7"; tail -6 ~/ber7/code/business_entity_resolution/logs/run7.log 2>/dev/null || echo "not started"
    echo "--- run 8"; tail -6 ~/ber8/code/business_entity_resolution/logs/run8.log 2>/dev/null || echo "not started"
    echo "--- EDA"; cat ~/analysis_logs/eda.log 2>/dev/null; tail -2 ~/analysis_logs/part_b.log 2>/dev/null
    echo "--- generator inference"; grep -c "" ~/analysis_logs/gen2.log 2>/dev/null | sed "s/^/log lines: /"; grep -aE "GEN2 DONE|Traceback" ~/analysis_logs/gen2.log
    echo "--- machine"; uptime; free -g | sed -n 2p' 2>&1
fi
echo "--- Kaggle (max 2 concurrent GPU sessions)"
for k in ber-biencoder ber-emb-test-india ber-emb-test-us ber-emb-test-france ber-emb-train-india ber-emb-train-us; do
  printf "  %-22s %s\n" $k "$($K kernels status rmuk16/$k 2>&1 | tail -1 | grep -oE 'RUNNING|QUEUED|COMPLETE|ERROR|CANCEL[A-Z_]*|404' | head -1)"
done
QP=$(cat ~/kaggle_queue.pid 2>/dev/null); echo "  queue driver: PID ${QP:-none} $(ps -p ${QP:-0} > /dev/null 2>&1 && echo running || echo NOT running), log ~/kaggle_queue.log"
tail -3 ~/kaggle_queue.log 2>/dev/null | sed 's/^/    /'
echo "  outputs landed: $(find submissions/emb -name 'nn_*.parquet' 2>/dev/null | wc -l) neighbour files in submissions/emb/"
LP=$(cat ~/launch_run8.pid 2>/dev/null); echo "--- run-8 launcher: PID ${LP:-none} $(ps -p ${LP:-0} > /dev/null 2>&1 && echo running || echo NOT running)"
tail -2 ~/launch_run8.log 2>/dev/null | sed 's/^/    /'
echo "--- submissions (S3)"
$A s3 ls s3://sagemaker-us-east-1-134051031272/ber/ --recursive 2>/dev/null | grep -E "submission|results|eda_results|run5p|run7|run8" | tail -12
echo "--- submissions (local)"; ls -la submissions/*.zip 2>/dev/null
