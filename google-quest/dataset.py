from torch.utils.data import Dataset


class QuestDataset(Dataset):
    
    def __init__(self, questions, answers, titles):
        
        self.questions = questions
        self.answers   = answers
        self.titles    = titles
        
    def __len__(self):
        return self.questions.shape[0]

    def __getitem__(self, idx):
        
        question = self.questions[idx]
        answer   = self.answers[idx]
        title    = self.titles[idx]
        
        return [question, answer, title]





