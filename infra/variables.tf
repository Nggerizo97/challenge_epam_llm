variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "challenge-epam-llm"
}

variable "bedrock_model_arns" {
  type = list(string)
  default = [
    "arn:aws:bedrock:us-east-1::foundation-model/meta.llama3-70b-instruct-v1:0",
    "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-text-express-v1",
  ]
}

variable "budget_limit_usd" {
  type    = number
  default = 10
}

variable "budget_email" {
  type        = string
  default     = ""
  description = "Optional. An email enables a Bedrock-only Budget alert."
}