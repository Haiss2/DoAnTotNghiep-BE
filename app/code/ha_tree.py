class HaTree:
    def __init__(self, fmc_):
        self.fmc_ = fmc_
        self.parent = [(0, self.fmc_), (self.fmc_, 1)]
        self.u_MPL= fmc_**2
        self.u_V = 1 - 3*fmc_**2
        self.haTree = self.parent
        for i in range(3):    
            self.haTree = self.a_step(self.haTree)

    def a_step(self, haTree):
        result = []
        for p in haTree:
            for i in range(4):
                _fmNode = p[0] + (p[1] - p[0])*i*self.u_MPL
                if (i == 3): 
                    __fmNode = _fmNode + (p[1] - p[0])*self.u_V
                else:
                    __fmNode = _fmNode + (p[1] - p[0])*self.u_MPL
                result.append((_fmNode, __fmNode))
        return result

    def getIndex(self, u):
        for i, (_fm, __fm) in enumerate(self.haTree):
            if _fm <= u and u < __fm:
                return i
        return(len(self.haTree))

# x = HaTree(0.4/0.6/2) 
# print(x.getIndex(0.4))


