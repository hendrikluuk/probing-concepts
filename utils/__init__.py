from utils.various import *

scoring_model = "gpt-4o"

# supply the URL of a proxy service that can evoke prompt templates with all models
# e.g. "https://subdomain.domain.com/api/service"
base_url = ""
# assume that there is a service to invoke predefined prompt templates
prompt_template_invocation_path = f"/template/"

# provide credentials
project = ""
user_id = ""

default_body = {
  "user_id": user_id,
  "project": project,
  "model": scoring_model
}
