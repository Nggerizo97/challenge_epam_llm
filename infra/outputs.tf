output "dynamodb_quota_table" {
  value = aws_dynamodb_table.llm_quota.name
}

output "streamlit_iam_user" {
  value = aws_iam_user.streamlit.name
}

output "streamlit_policy_arn" {
  value = aws_iam_policy.streamlit.arn
}