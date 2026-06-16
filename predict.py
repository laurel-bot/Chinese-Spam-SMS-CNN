import torch
import pickle
import os
from dataset import clean_text, tokenize
from model import TextCNN

class SpamPredictor:
    def __init__(self, model_path='best_model.pth', vocab_path='vocab.pkl', max_len=100):
        self.max_len = max_len
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters (must match train.py config)
        embedding_dim = 128
        num_filters = 100
        filter_sizes = [3, 4, 5]
        
        self.model = TextCNN(len(self.vocab.word2idx), embedding_dim, num_filters, filter_sizes, num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        tokens = tokenize(text)
        indices = self.vocab.transform(tokens, self.max_len)
        tensor_input = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor_input)
            probs = torch.softmax(outputs, dim=1).squeeze(0)
            pred_class = torch.argmax(probs).item()
            
        return pred_class, probs[pred_class].item()

if __name__ == '__main__':
    predictor = None
    if not os.path.exists('best_model.pth') or not os.path.exists('vocab.pkl'):
        print("Please run train.py first to train the model and generate 'best_model.pth' and 'vocab.pkl'.")
    else:
        predictor = SpamPredictor()
        
        test_cases = [
            "【天上掉下个Iphone5！】回对应数字设置优先接收的回执精品资讯，可参与抽奖。",
            "尊敬的旅客，您在官网的机票订单已支付成功，祝您旅途愉快。"
        ]
        
        for text in test_cases:
            pred, prob = predictor.predict(text)
            label_str = "Spam (垃圾短信)" if pred == 1 else "Ham (正常短信)"
            print(f"Text: {text}\nPrediction: {label_str} | Confidence: {prob:.4f}\n")
