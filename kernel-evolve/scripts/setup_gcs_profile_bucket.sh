#!/usr/bin/env bash
# One-time setup for GCS profile artifacts bucket and IAM bindings.
# Run this manually with appropriate gcloud credentials.
set -euo pipefail

PROJECT="tpu-service-473302"
BUCKET="glaucis-profiles"
REGION="us-central1"
GCP_SA="kernel-eval"
K8S_SA="kernel-eval"
K8S_NS="default"
# Update this with your actual GKE cluster name
GKE_CLUSTER="tpu7x-cluster"

echo "=== Creating GCS bucket ==="
gcloud storage buckets create "gs://${BUCKET}" \
  --project="${PROJECT}" \
  --location="${REGION}" \
  --uniform-bucket-level-access \
  2>/dev/null || echo "Bucket already exists"

echo "=== Setting lifecycle (7-day auto-delete) ==="
cat <<'LIFECYCLE' | gcloud storage buckets update "gs://${BUCKET}" --lifecycle-file=-
{"rule": [{"action": {"type": "Delete"}, "condition": {"age": 7}}]}
LIFECYCLE

echo "=== Creating GCP service account ==="
gcloud iam service-accounts create "${GCP_SA}" \
  --project="${PROJECT}" \
  --display-name="Kernel Eval Pod SA" \
  2>/dev/null || echo "SA already exists"

echo "=== Granting GCS objectCreator to SA ==="
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${GCP_SA}@${PROJECT}.iam.gserviceaccount.com" \
  --role="roles/storage.objectCreator"

echo "=== Creating K8s service account ==="
kubectl create serviceaccount "${K8S_SA}" -n "${K8S_NS}" \
  2>/dev/null || echo "K8s SA already exists"

echo "=== Binding K8s SA to GCP SA via Workload Identity ==="
gcloud iam service-accounts add-iam-policy-binding \
  "${GCP_SA}@${PROJECT}.iam.gserviceaccount.com" \
  --role=roles/iam.workloadIdentityUser \
  --member="principal://iam.googleapis.com/projects/785128357837/locations/global/workloadIdentityPools/${GKE_CLUSTER}.svc.id.goog/subject/ns/${K8S_NS}/sa/${K8S_SA}"

echo "=== Annotating K8s SA ==="
kubectl annotate serviceaccount "${K8S_SA}" -n "${K8S_NS}" \
  "iam.gke.io/gcp-service-account=${GCP_SA}@${PROJECT}.iam.gserviceaccount.com" \
  --overwrite

echo "=== Done! Verify with: ==="
echo "kubectl get sa ${K8S_SA} -n ${K8S_NS} -o yaml"
