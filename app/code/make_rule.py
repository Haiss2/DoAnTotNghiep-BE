class Rule:
    def __init__(self, data, cluster, num_classes, num_rules):
        self.data = data
        self.cluster = cluster
        self.rules = []
        self.e_rule = 0.01
        self.num_classes = num_classes
        self.num_rules = num_rules
        self.num_properties = len(self.data[0]) - 1
        for (ix, d) in enumerate(self.data):
            for i in range(2**self.num_properties):
                s = []
                b = '0'*(self.num_properties - len(bin(i)[2:])) + bin(i)[2:]
                u_ = 1
                for j in range(self.num_properties):
                    burn_set = self.burn(d[j+1], self.cluster[j])[int(b[j])]
                    if u_ < self.e_rule:
                        break
                    s.append(burn_set[0])
                    u_ *= burn_set[1]
                if u_ > self.e_rule and 0.05 > u_:
                    self.add_rule([d[0]] + s + [u_])

    def do_thuoc(self, x, c1, c2, c3):
        if x >= c3 or x <= c1: 
            return 0
        if c1 < x and x <= c2:
            return (x - c1) / (c2 - c1)
        if c2 < x and x < c3:
            return (c3 - x) / (c3 - c2)

    def burn(self, x, c):
        for id, pt in enumerate(c + [100]):
            if x <= pt:
                break
        if id > 0 and id < len(c) - 1:
            u = self.do_thuoc(x, c[id - 1], c[id], c[id+1])
            return [(id - 1, 1 - u), (id, u)]
        if id == 0:
            return [(id, self.do_thuoc(x, 0, c[0], c[1] )), (0, 0)]
        if id == len(c):
            return [(id - 1, self.do_thuoc(x, c[id - 2], c[id -1], 100)), (0, 0)]
        if id == len(c) - 1:
            u = self.do_thuoc(x, c[id - 1], c[id], 100)
            return [(id - 1, 1 - u), (id, u)]


    def add_rule(self, rule):
        loai = False
        for r in self.rules:
            if r[1:self.num_properties + 1] == rule[1:self.num_properties + 1]:
                loai = True
                if r[self.num_properties] < rule[self.num_properties]:
                    r = rule
        if not loai:
            self.rules.append(rule)

    def colection_rules(self):
        percent = [0 for i in range(self.num_classes)]
        for i in self.data:
            percent[int(i[0])-1] += 1
        a = [(i/sum(percent))**1 for i in percent]
        b = [int(i*self.num_rules) for i in a]
        def GetBurn(rule):
            return rule[self.num_properties + 1]
        self.rules.sort(reverse=True, key=GetBurn)
        _rules = []
        for rule in self.rules:
            if b[int(rule[0]) - 1] > 0:
                _rules.append(rule)
                b[int(rule[0]) - 1] -= 1
        def SortByClass(rule):
            return rule[0]
        _rules.sort(key=SortByClass)
        return _rules