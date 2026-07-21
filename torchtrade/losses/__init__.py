from torchtrade.losses.ctrl import CTRLLoss, CTRLPPOLoss
from torchtrade.losses.dg_loss import DGLoss
from torchtrade.losses.group_relative_pg_loss import GroupRelativePGLoss

# SAOLoss (torchtrade.losses.sao_loss) is intentionally NOT imported here: it
# subclasses torchrl's LLM GRPOLoss and pulls the heavy LLM/vllm stack. Import it directly
# or use it via LLMTrainer(loss="sao").
