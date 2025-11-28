from models import MultimodalSentimentModel
from colorama import Fore


def count_parameters(model):
    """
    Count parameters per component and total.
    Returns: (total_params, params_dict)
    """
    params_dict = {
        'text_encoder': 0,
        'video_encoder': 0,
        'audio_encoder': 0,
        'fusion_layer': 0,
        'emotion_classifier': 0,
        'sentiment_classifier': 0
    }

    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.numel()
            total_param += n

            if 'text_encoder' in name:
                params_dict['text_encoder'] += n
            elif 'video_encoder' in name:
                params_dict['video_encoder'] += n
            elif 'audio_encoder' in name:
                params_dict['audio_encoder'] += n
            elif 'fusion_layer' in name:
                params_dict['fusion_layer'] += n
            elif 'emotion_classifier' in name:
                params_dict['emotion_classifier'] += n
            elif 'sentiment_classifier' in name:
                params_dict['sentiment_classifier'] += n

    return total_param, params_dict


if __name__ == "__main__":
    model = MultimodalSentimentModel()
    total_params, param_dict = count_parameters(model=model)

    print("Parameter count by component")
    for component, count in param_dict.items():
        print(Fore.CYAN + f"{component:22s}:" +
              Fore.RESET + f" {count:,} parameters")

    print(Fore.GREEN +
          f"\nTotal trainable parameters: {total_params:,}" + Fore.RESET)
