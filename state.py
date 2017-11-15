
class state_instance:
    def __init__(self, inst):
        # self.hyperParams = hyperParams
        self.chars = inst.chars
        self.gold = inst.gold
        self.words = []
        self.pos_id = []
        self.pos_labels = []
        self.actions = []

        self.word_hiddens = []
        self.word_cells = []

        self.all_h = []
        self.all_c = []


