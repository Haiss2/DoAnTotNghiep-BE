class Predict:
    def __init__(self, num_properties, cluster, rule, k_mean):
        self.cluster = cluster
        self.rules = rule
        self.num_properties = num_properties
        self.k_mean = k_mean

    def do_thuoc(self, x, c1, c2, c3):
        if x >= c3 or x <= c1: 
            return 0
        if c1 < x and x <= c2:
            return (x - c1) / (c2 - c1)
        if c2 < x and x < c3:
            return (c3 - x) / (c3 - c2)

    # x = data, s = index of Cluseter, C = cluster of attribute
    def do_thuoc_set(self, x, s, C):
        if s == 0:
            return self.do_thuoc(x, 0, C[0], C[1])
        if s == self.k_mean - 1:
            return self.do_thuoc(x, C[s - 1], C[s], 100)
        return self.do_thuoc(x, C[s-1], C[s], C[s+1])

    def predict(self, t, r): # t la ban ghi
        rule_burn = [min( [self.do_thuoc_set(t[i+1], rule[i+1], self.cluster[i]) for i in range(self.num_properties)]) for rule in r]
        do_thuoc_sum = [sum( [self.do_thuoc_set(t[i+1], rule[i+1], self.cluster[i])**0.5 for i in range(self.num_properties)]) for rule in r]
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
            "rule": r[id][1:self.num_properties + 1],
            "sum": len(ids)
        }

    def get_rule_truth(self, t): # t la ban ghi
        sub_rules = []
        get_first = False
        first_id = 0
        for i, j in enumerate(self.rules):
            if int(j[0]) == int(t[0]):
                sub_rules.append(j)
                if not get_first:
                    first_id = i
                    get_first = True
        correct_rule = self.predict(t, sub_rules)
        return {
            "predict": t[0],
            "rule_id": correct_rule["rule_id"] + first_id,
            "rule": correct_rule["rule"]
        }
    
    
    def detect_cluster(self, a, b, data):
        result = []
        for i in range(self.num_properties):
            if a[i] != b[i] and abs(a[i] - b[i]) == 1:
                result += [[i, a[i], self.do_thuoc_set(data[i], a[i], self.cluster[i]), b[i], self.do_thuoc_set(data[i], b[i], self.cluster[i])]]
        return result
