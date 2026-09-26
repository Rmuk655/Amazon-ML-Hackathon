#!/bin/bash
# Laptop-side launcher for run 8 (detached; survives the Claude session):
#   setsid nohup code/business_entity_resolution/launch_run8.sh > /dev/null 2>&1 &
# PID ~/launch_run8.pid, log ~/launch_run8.log.
# Waits until (a) run 7 has finished (instance stopped and run7 artifacts in S3) and (b) the Kaggle bi-encoder
# outputs for BOTH splits have landed in submissions/emb/ (queue driver). Then starts the instance, uploads the
# current code to ~/ber8 and the neighbour files to ~/embed, and runs run8.sh (which stops the instance at the end).
# Gives up (no launch) if the embeddings are not complete by CUTOFF - run 8 would only duplicate run 7.
cd "$(dirname "$0")/../.."
ROOT=$(pwd); A=$ROOT/venv/bin/aws; PY=$ROOT/venv/bin/python; KEY=~/.ssh/claude-ber.pem; I=i-0076c2fcc9a8df3ab
S3=s3://sagemaker-us-east-1-134051031272/ber; EMB=$ROOT/submissions/emb
CUTOFF=${CUTOFF:-"2026-09-27 11:00"}
echo $$ > ~/launch_run8.pid; exec >> ~/launch_run8.log 2>&1
log() { echo "[$(date '+%a %H:%M')] $*"; }
emb_ready() {   # neighbour files for both splits (either the all-in-one job or the per-country backups)
  local n_test n_train
  n_test=$(find $EMB -name 'nn_test_*.parquet' 2>/dev/null | wc -l); n_train=$(find $EMB -name 'nn_train_*.parquet' 2>/dev/null | wc -l)
  [ "$n_test" -ge 6 ] && [ "$n_train" -ge 4 ]
}
run7_done() {
  [ "$($A ec2 describe-instances --instance-ids $I --query 'Reservations[0].Instances[0].State.Name' --output text)" = stopped ] \
    && $A s3 ls $S3/run7-2026-09-26/run7_logs.tgz > /dev/null 2>&1
}
log "launcher started (cutoff $CUTOFF)"
until run7_done; do sleep 300; done
log "run 7 finished"
until emb_ready; do
  [ "$(date +%s)" -gt "$(date -d "$CUTOFF" +%s)" ] && { log "embeddings not complete by $CUTOFF -> run 8 NOT launched"; exit 0; }
  sleep 300
done
log "embeddings ready: $(find $EMB -name 'nn_*.parquet' | wc -l) files"
$A ec2 start-instances --instance-ids $I > /dev/null && $A ec2 wait instance-running --instance-ids $I
IP=$($A ec2 describe-instances --instance-ids $I --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
for i in $(seq 1 30); do ssh -i $KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new ubuntu@$IP true 2>/dev/null && break; sleep 5; done
TMP=$(mktemp -d)
tar -czf $TMP/code.tgz --exclude=__pycache__ --exclude=logs code/business_entity_resolution Documentation_template.md
tar -czf $TMP/embed.tgz -C $EMB $(cd $EMB && find . -name 'nn_*.parquet')
$PY - > $TMP/urls.env <<'EOF'
import boto3
s3 = boto3.client("s3", region_name="us-east-1")
for n, k in (("URL_sub8", "ber/run8-2026-09-27/submission8.zip"), ("URL_run8_logs", "ber/run8-2026-09-27/run8_logs.tgz")):
    print(f"{n}='" + s3.generate_presigned_url("put_object", Params={"Bucket": "sagemaker-us-east-1-134051031272", "Key": k}, ExpiresIn=3 * 24 * 3600) + "'")
EOF
scp -q -i $KEY $TMP/code.tgz $TMP/embed.tgz $TMP/urls.env ubuntu@$IP:~/ && ssh -i $KEY ubuntu@$IP '
  mkdir -p ~/ber8 ~/embed && tar -xzf ~/code.tgz -C ~/ber8 && tar -xzf ~/embed.tgz -C ~/embed
  ln -sfn ~/ber/dataset ~/ber8/dataset; ln -sfn ~/ber/utils ~/ber8/utils; ln -sfn ~/ber/venv ~/ber8/venv
  cat ~/urls.env >> ~/run5_urls.env; sudo shutdown -h +540 "cost guard run8"
  setsid nohup bash ~/ber8/code/business_entity_resolution/run8.sh > ~/run8.out 2>&1 < /dev/null & disown'
log "run 8 launched on $IP"
rm -rf $TMP
