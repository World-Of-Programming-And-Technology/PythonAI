import unittest
from core import predict_entities, model, vectorizer, FrameClassifier
import torch

class TestCoreFunctions(unittest.TestCase):

    def test_predict_entities_name(self):
        sentence = "My name is Aryan"
        result = predict_entities(sentence)
        self.assertEqual(result['NAME'], "Aryan")
    
    def test_predict_entities_location(self):
        sentence = "I live in Jaipur"
        result = predict_entities(sentence)
        self.assertEqual(result['LOCATION'], "Jaipur")

    def test_frame_prediction_question(self):
        input_sentence = "Where do you live?"
        vectorized = vectorizer.transform([input_sentence]).toarray()
        input_tensor = torch.tensor(vectorized, dtype=torch.float32)
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()
        self.assertEqual(predicted, 1)  # 1 = Question

    def test_frame_prediction_command(self):
        input_sentence = "Close the window"
        vectorized = vectorizer.transform([input_sentence]).toarray()
        input_tensor = torch.tensor(vectorized, dtype=torch.float32)
        output = model(input_tensor)
        predicted = torch.argmax(output, dim=1).item()
        self.assertEqual(predicted, 2)  # 2 = Command

if __name__ == '__main__':
    unittest.main()
