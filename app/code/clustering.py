class Clustering:
    def __init__(self, data, k_mean):
        self.data = data
        self.k_mean = k_mean
        self.m_fuzzy = 2
        self.num_properties = len(self.data[0]) - 1
        self.clusters = []
        self.e_fcm = 0.001
        for x in range(self.num_properties):
            o, c, u = self.step1()
            while not self.step4(o, c):
                o = c.copy()
                self.step2(x+1, c, u)
                self.step3(x+1, c, u)
            c.sort()
            self.clusters.append(c)
    
    def step1(self):
        o = [0 for i in range(self.k_mean)]
        c = [i*100/self.k_mean for i in range(self.k_mean)]
        u  = [ o.copy() for i in range(len(self.data))]
        return o, c, u

    def step2(self, x, c, u):
        power = 2/(self.m_fuzzy - 1)
        for i in range(len(self.data)):
            for j in range(self.k_mean):
                xi = self.data[i][x]
                mau = sum([(abs(xi - c[j]) / abs( 0.00001 if xi - c[k] == 0 else xi - c[k]) )** power for k in range(self.k_mean)])
                u[i][j] = 1/( 0.00001 if mau == 0 else mau)

    def step3(self, x, c, u):
        for j in range(self.k_mean):
            tu = sum([u[i][j]**self.m_fuzzy * self.data[i][x] for i in range(len(self.data))])
            mau  = sum([u[i][j]**self.m_fuzzy for i in range(len(self.data))])
            c[j] = tu/mau

    def step4(self, old, new):
        result = True
        for i in range(self.k_mean):
            result = result and abs(old[i] - new[i]) < self.e_fcm
        return result