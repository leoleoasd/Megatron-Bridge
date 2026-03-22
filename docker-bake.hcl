group "default" {
  targets = ["megatron_bridge"]
}

target "megatron_bridge" {
  context = "."
  dockerfile = "Dockerfile"
  args = {
    REPO = "248189905876.dkr.ecr.us-east-1.amazonaws.com/greenland"
    BASE_TAG = "base"
  }
  tags = ["248189905876.dkr.ecr.us-east-1.amazonaws.com/greenland:megatron_bridge"]
  platforms = ["linux/amd64"]
  push = true
}
