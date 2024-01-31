from configuration_chatglm import ChatGLMConfig

from transformers import PreTrainedModel
import torchvision.transforms as T

class ChatGLMForConditionalGenerationWithImage(PreTrainedModel):
    def __init__(self, config: ChatGLMConfig, empty_init=True):
        super().__init__(config, empty_init=empty_init)
        from .visual import BLIP2
        self.image_encoder = BLIP2(config.eva_config, config.qformer_config)
    
    def forward(self, img):
        img = img.transpose(1, 2).transpose(1, 3).contiguous() / 255.0
        images = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))(img)
        image_embeds = self.image_encoder(images)
        return image_embeds