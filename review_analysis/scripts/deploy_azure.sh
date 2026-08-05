#!/usr/bin/env bash
# Build this folder and deploy it to the existing Azure Web App.
#
# Defaults target the app in the portal screenshot:
#   Web App   proforma-demo-2   (Linux container, P0v3)
#   RG        son_eastus2_proforma_rg02
#   Registry  proformaacr.azurecr.io
#
# It pushes a NEW repository (review-analysis) rather than overwriting
# proforma-demo-2:latest, so the image that app runs today stays intact and
# rolling back is one command (see the end of this file).
#
# Usage:
#   export AZURE_OPENAI_API_KEY=...          # optional; enables the AI button
#   ./scripts/deploy_azure.sh                # build, push, point the app at it
#   IMAGE_TAG=v2 ./scripts/deploy_azure.sh   # pin your own tag
#
# Requires: docker (with buildx) and az, and `az login` already done.

set -euo pipefail

APP_NAME="${APP_NAME:-proforma-demo-2}"
RESOURCE_GROUP="${RESOURCE_GROUP:-son_eastus2_proforma_rg02}"
ACR_NAME="${ACR_NAME:-proformaacr}"
IMAGE_REPO="${IMAGE_REPO:-review-analysis}"
IMAGE_TAG="${IMAGE_TAG:-$(date +%Y%m%d-%H%M%S)}"

REGISTRY="${ACR_NAME}.azurecr.io"
IMAGE="${REGISTRY}/${IMAGE_REPO}:${IMAGE_TAG}"
APP_PORT=8501

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

echo "==> Building ${IMAGE}"
# --platform linux/amd64 is not optional on an Apple-silicon Mac: App Service
# runs x86_64, and an arm64 image starts and then dies with an exec-format
# error that shows up only in the container log.
docker buildx build --platform linux/amd64 -t "$IMAGE" -t "${REGISTRY}/${IMAGE_REPO}:latest" --load .

echo "==> Pushing to ${REGISTRY}"
az acr login --name "$ACR_NAME"
docker push "$IMAGE"
docker push "${REGISTRY}/${IMAGE_REPO}:latest"

echo "==> Pointing ${APP_NAME} at the new image"
az webapp config container set \
  --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --container-image-name "$IMAGE" \
  --container-registry-url "https://${REGISTRY}"

echo "==> App settings"
# WEBSITES_PORT tells App Service which port the container listens on; without
# it the platform probes 8080 and the site returns 504 while the app is fine.
settings=(
  "WEBSITES_PORT=${APP_PORT}"
  "WEBSITES_CONTAINER_START_TIME_LIMIT=600"
  "REVIEW_ANALYSIS_DEMO_LOCKED=${REVIEW_ANALYSIS_DEMO_LOCKED:-1}"
  "AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT:-https://soneastus2proformaai.openai.azure.com}"
  "AZURE_OPENAI_DEPLOYMENT=${AZURE_OPENAI_DEPLOYMENT:-gpt-4o}"
)
if [[ -n "${AZURE_OPENAI_API_KEY:-}" ]]; then
  settings+=("AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}")
else
  echo "    (no AZURE_OPENAI_API_KEY in the environment — leaving the existing"
  echo "     app setting alone; the dashboard runs either way)"
fi
az webapp config appsettings set \
  --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --settings "${settings[@]}" --output none

echo "==> Web sockets + health check"
# Streamlit talks to the browser over a websocket; without this the page loads
# and then sits on "connecting".
az webapp config set \
  --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --web-sockets-enabled true --output none
az webapp config set \
  --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --generic-configurations '{"healthCheckPath": "/_stcore/health"}' --output none
# One instance today, but sticky sessions keep a session on one container if
# the plan is ever scaled out.
az webapp update \
  --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --client-affinity-enabled true --output none

echo "==> Restarting"
az webapp restart --name "$APP_NAME" --resource-group "$RESOURCE_GROUP"

url="https://$(az webapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" \
  --query defaultHostName -o tsv)"
echo
echo "Deployed ${IMAGE}"
echo "URL:     ${url}"
echo
echo "Watch it come up:"
echo "  az webapp log tail --name ${APP_NAME} --resource-group ${RESOURCE_GROUP}"
echo
echo "Roll back to the previous app:"
echo "  az webapp config container set --name ${APP_NAME} --resource-group ${RESOURCE_GROUP} \\"
echo "    --container-image-name ${REGISTRY}/proforma-demo-2:latest \\"
echo "    --container-registry-url https://${REGISTRY}"
