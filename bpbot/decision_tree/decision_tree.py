import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.cluster import KMeans
import graphviz 
np.set_printoptions(suppress=True)

class DecisionTree(object):
    def __init__(self):
        self.clf = DecisionTreeClassifier(max_depth=3, criterion='entropy')

        self.est = KMeans(n_clusters=3)#构造聚类器
        init_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dt_ini.csv")
        self.add_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dt_add.csv")
        add = np.loadtxt(self.add_path)
        if add.shape[0]==0:
            self.out = np.loadtxt(init_path) 
        else:
            self.out = np.r_[np.loadtxt(init_path), add if len(add.shape) > 1 else [add]]
        self.X_ = np.abs(self.out[:,[0,1,2,5]])
        self.y_ = self.out[:,-1]
        
        self.train(self.X_, self.y_)
        
    def train(self, X, y):
        self.X_ = X
        self.y_ = y
        self.clf.fit(X,y)
        # self.est.fit(self.out[:,[2,5]])#聚类
        # label_pred = self.est.labels_ #获取聚类标签
        # centroids = self.est.cluster_centers_ #获取聚类中心
        # inertia = self.est.inertia_ # 获取聚类准则的总和
        # print(label_pred)
        # print(y)

    def save(self, f, a):
        add_ = np.append(f, a)
        np.savetxt(self.add_path, np.r_[self.out, [add_]])
    
    def finetune(self, X, y):
        """
        Args:
            X (N,num_attributes)_
            y (N,)
        """
        self.X_ = np.r_[self.X_, X]
        self.y_ = np.r_[self.y_, y]
        self.clf.fit(self.X_, self.y_)

    def infer(self, X):
        return self.clf.predict(X) 

    def expectation(self, M):
        m = []
        for i in range(M.shape[1]):
            a = M[:,i].tolist()
            a.remove(max(a))
            a.remove(min(a))
            m.append(np.mean(a))
        return np.array(m)
    
    def plot_data(self):
        X = self.X_
        y = self.y_
        plt.scatter(X[y==0,0],X[y==0,1], color='g')
        plt.scatter(X[y==1,0],X[y==1,1], color='r')
        plt.scatter(X[y==2,0],X[y==2,1], color='r', facecolors='none')
        plt.legend(['Success', 'Fling', 'Regrasp'])
        plt.xlabel("Fz")
        plt.ylabel("Mz")
        plt.show()
    
    def plot_tree(self):
        # dot_data = export_graphviz(self.clf, out_file=None) 
        # graph = graphviz.Source(dot_data) 
        # graph.render("dt") 
        r = export_text(self.clf, feature_names=['Fz','Mz'])
        print(r)

    def render_tree(self):
        dot_data = export_graphviz(self.clf, out_file=None, 
                             feature_names=['Fz','Mz'],  
                             class_names=['Success,','Fling','Regrasp'],  
                             filled=True, rounded=True,  
                             special_characters=True)  
        graph = graphviz.Source(dot_data)  
        graph.render("dt_vis")

# dt = DecisionTree()
# dt.plot_data()
# dt.save([1,2,3,4,5,6], 999)


# dt.finetune([[1.0,0.3444]], [0])
# dt.plot_data()


# import glob
# def expectation(M):
#     m = []
#     for i in range(M.shape[1]):
#         a = M[:,i].tolist()
#         a.remove(max(a))
#         a.remove(min(a))
#         m.append(np.mean(a))
#     return np.array(m)
# X = []
# y = []
# for f in glob.glob("/home/hlab/Desktop/dt_ini_data/*"):
#     if "success" in f: #0
#         mean = expectation(np.loadtxt(f))
#         X.append(mean)
#         y.append(0)
#     elif "fling" in f: #1
#         mean = expectation(np.loadtxt(f))
#         X.append(mean)
#         y.append(1)
#     elif "regrasp" in f: #2
#         mean = expectation(np.loadtxt(f))
#         X.append(mean)
#         y.append(2)
# X = np.array(X)[:,[2,5]]
# y = np.array(y)
# print(X.shape, y.shape)

# plt.scatter(X[y==0,0],X[y==0,1], color='g')
# plt.scatter(X[y==1,0],X[y==1,1], color='r')
# plt.scatter(X[y==2,0],X[y==2,1], color='r', facecolors='none')
# plt.legend(['Success', 'Fling', 'Regrasp'])
# plt.xlabel("Fz")
# plt.ylabel("Mz")
# plt.show()

# dt_clf = DecisionTreeClassifier(max_depth=3, criterion='entropy')
# dt_clf.fit(X,y)
# print("Trained")

import tkinter as tk 
class DecisionWindow(object):
    def __init__(self):
        self.master=tk.Tk()
        self.master.geometry("400x300")
        self.master.title("Human Superviion")
        self.show_widgets()
        self.act = -1

    def click_button(self, act):
        self.master.destroy()
        self.act = act

    def show_widgets(self):
        self.frame = tk.Frame(self.master)
        self.button0 = self.create_button("Success", lambda: self.click_button(0))
        self.button1 = self.create_button("Fling", lambda: self.click_button(1))
        self.button2 = self.create_button("Regrasp", lambda: self.click_button(2))

        self.frame.pack()
    
    def create_button(self, text, command):
        butt = tk.Button(
            self.frame,
            border=1,
            relief="ridge",
            compound=tk.CENTER,
            text=text,
            fg="black",
            font="Arial 36",
            command=command)
        butt.pack()
        return butt
    


# app = DecisionWindow()
# app.master.mainloop()
# print(app.ret)
# print(supervised_act)

# sample.py
