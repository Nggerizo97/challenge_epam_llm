terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

resource "aws_dynamodb_table" "llm_quota" {
  name         = "${var.project_name}-quota"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "quota_key"

  attribute {
    name = "quota_key"
    type = "S"
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }

  server_side_encryption {
    enabled = true
  }

  tags = {
    Project = var.project_name
    Purpose = "Persistent LLM cost quota"
  }
}

resource "aws_iam_user" "streamlit" {
  name = "${var.project_name}-streamlit"
}

resource "aws_iam_policy" "streamlit" {
  name        = "${var.project_name}-streamlit-inference"
  description = "Allows this Streamlit app to reserve quota and invoke approved Bedrock models only."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["dynamodb:UpdateItem"]
        Resource = aws_dynamodb_table.llm_quota.arn
      },
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = var.bedrock_model_arns
      }
    ]
  })
}

resource "aws_iam_user_policy_attachment" "streamlit" {
  user       = aws_iam_user.streamlit.name
  policy_arn = aws_iam_policy.streamlit.arn
}

resource "aws_budgets_budget" "bedrock" {
  count = var.budget_email == "" ? 0 : 1

  name         = "${var.project_name}-bedrock-monthly"
  budget_type  = "COST"
  limit_amount = tostring(var.budget_limit_usd)
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_filter {
    name   = "Service"
    values = ["Amazon Bedrock"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }
}