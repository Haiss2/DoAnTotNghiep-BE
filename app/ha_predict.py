from app import config as cf
from app.code.ha_tree import HaTree

class HAPredict:
    def __init__(self, data, cluster, rule, fmc, k_mean):
        self.data = data
        self.cluster = cluster
        self.rules = rule
        self.num_properties = len(self.data[0]) - 1
        self.fmc = fmc
        self.k_mean = k_mean
        self.ha_trees = [[HaTree(fmc_) for fmc_ in line ] for line in self.fmc]
    def do_thuoc(self, x, c1, c2, c3, cluterIndex, attributeIndex):
        if x >= c3 or x <= c1: 
            return 0
        if c1 < x and x <= c2:
            return self.ha_trees[attributeIndex][2*cluterIndex].getIndex((x - c1) / (c2 - c1))
        if c2 < x and x < c3:
            return self.ha_trees[attributeIndex][2*cluterIndex+1].getIndex((c3 - x) / (c3 - c2))

    def do_thuoc_set(self, x, s, C, attribute):  
        if s == 0:
            return self.do_thuoc(x, 0, C[0], C[1], s, attribute)
        if s == self.k_mean - 1:
            return self.do_thuoc(x, C[s - 1], C[s], 100, s, attribute)
        return self.do_thuoc(x, C[s-1], C[s], C[s+1], s, attribute)

    def predict(self, t, r): # t la ban ghi
        arr = [ [self.do_thuoc_set(t[i+1], rule[i+1], self.cluster[i], i) for i in range(self.num_properties)] for rule in r]
        
        rule_burn = [min(a) for a in arr]
        do_thuoc_sum = [sum([x**0.5 for x in a]) for a in arr]
        maxx = max(rule_burn)
        summ = 0
        ids = []
        for (i, j) in enumerate(rule_burn):
            if j == maxx:
                ids.append(i)
        if len(ids) == 1:
            id = rule_burn.index(maxx)
        else:
            summ = [do_thuoc_sum[i] for i in ids]
            id = do_thuoc_sum.index(max(summ))
        return {
            "predict": int(r[id][0]),
            "rule_id": id,
            "rule": r[id][1: self.num_properties +1]
        }
    def num_corrects(self):
        corrects = 0
        for ix, record in enumerate(self.data):
            if int(record[0]) == self.predict(record, self.rules)["predict"]:
                corrects += 1
        return corrects


