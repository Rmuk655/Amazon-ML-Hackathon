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
    echo "--- run 5 (overnight)"; tail -4 $L/run5.log 2>/dev/null || echo "not started"
    echo "--- EDA"; cat ~/analysis_logs/eda.log 2>/dev/null; tail -2 ~/analysis_logs/part_b.log 2>/dev/null
    echo "--- generator inference"; grep -c "" ~/analysis_logs/gen2.log 2>/dev/null | sed "s/^/log lines: /"; grep -aE "GEN2 DONE|Traceback" ~/analysis_logs/gen2.log
    echo "--- machine"; uptime; free -g | sed -n 2p' 2>&1
fi
echo "--- Kaggle"
$K kernels status rmuk16/ber-biencoder 2>&1 | tail -1
echo "--- submissions (S3)"
$A s3 ls s3://sagemaker-us-east-1-134051031272/ber/ --recursive 2>/dev/null | grep -E "submission|results|eda_results|run5p" | tail -10
echo "--- submissions (local)"; ls -la submissions/*.zip 2>/dev/null
