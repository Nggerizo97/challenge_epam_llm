# Point the app, the AWS CLI and boto3 at a local AWS Emulator.
# Usage:  source scripts/emulator_env.sh [port]     (default port 9297)
EMULATOR_PORT="${1:-9297}"
export AWS_ENDPOINT_URL="https://127.0.0.1:${EMULATOR_PORT}"
export AWS_SHARED_CREDENTIALS_FILE="$HOME/.aws/emulator/${EMULATOR_PORT}/emulator_credentials"
export AWS_CONFIG_FILE="$HOME/.aws/emulator/${EMULATOR_PORT}/config"
export AWS_CA_BUNDLE="$HOME/.aws/emulator/${EMULATOR_PORT}/ca.pem"
export AWS_EC2_METADATA_DISABLED=true
export AWS_DISABLE_HOST_PREFIX_INJECTION=true
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1
# Real AWS credentials in the environment would override the emulator's keys.
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_PROFILE
