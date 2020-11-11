from tkinter import *
import cv2 as cv
from PIL import Image,ImageTk
import numpy as np
import tkinter.filedialog as filedialog
from keras import models
from model.BP_test import BPNetwork
from sklearn.externals import joblib


class window():
    root:Tk
    # 图像处理时初始图片
    im_src:np.array = None
    # 存储处理后的图像
    im_dist:list = []
    # 仅供显示的图像，会将图像改变到合适的大小
    im_show:np.array = None
    im_tk:PhotoImage
    menubar:Menu
    imageforshow:Canvas
    file_path_open:str
    width:int
    height:int
    start_point:tuple
    end_point:tuple
    rect = None
    cursor_index:int = 1
    src_size_changed:bool = True
    startX:int
    startY:int
    endX:int
    endY:int
    model_type:str = "default"
    def __init__(self):
        self.im_src = cv.imread("./src/Altair_dagger.jpg", cv.COLOR_BGR2GRAY)
        self.initial()
        self.root.mainloop()
# 初始化
    def initial(self):
        self.root = Tk()
        self.root.title("数据挖掘实验课作业")
        self.width_max,self.height_max=self.root.maxsize()
        self.root.geometry("%sx%s"%(int(self.width_max/2),int(self.height_max/2)))
        self.initial_menu()
        # self.imLable.pack()
        # 为左侧图像绑定鼠标草做
        self.imageforshow = Canvas(self.root)
        self.imageforshow.bind("<ButtonPress-1>", self.start_select)
        self.imageforshow.bind("<B1-Motion>", self.dynamic_rect)
        self.imageforshow.bind("<ButtonRelease-1>", self.delete_rect)

        # 设置右侧显示区域
        self.right_area=Frame(self.root)
        self.right_area.pack(side=RIGHT,fill=Y)
        # 在右侧区域放置显示标签内容的文本框
        self.show_text = Text(self.right_area,height = 5,width = 40)
        self.show_text.insert(INSERT, "正在使用标准模型")
        self.show_text.pack()
        self.predict_text = Text(self.right_area,height = 2,width =40)
        self.predict_text.pack()
        # 设置右下方固定大小的图像显示区域
        self.im_area = Frame(self.right_area,height = 200,width = 200,bg = 'gray')
        self.im_area.pack(side=BOTTOM)
        # 在右侧区域放置画布
        self.reception_field = Canvas(self.im_area)
        self.imageforshow.pack()
        self.reception_field.pack()
        self.root.config(menu=self.menubar)
        # 导入默认模型
        self.model = joblib.load('model/模型/BP.pkl')  # 读取模型
# 初始化菜单按钮，并绑定处理点击时间的函数
    def initial_menu(self):
        self.menubar = Menu(self.root)
        #创建文件子菜单
        self.menu_file=Menu(self.menubar)
        self.menu_file.add_command(label="导入图像",command=self.open_im_dir)
        self.menu_file.add_command(label="导入模型",command=self.open_model_dir)
        self.menubar.add_cascade(label="文件",menu=self.menu_file)

# 点击响应
    # 绘图点击事件响应
    def start_select(self,event):
        self.startX,self.startY = (event.x, event.y)
        self.rect = self.imageforshow.create_rectangle(event.x,event.y,max(event.x,0),max(event.y,0),fill="")
    def dynamic_rect(self, event):
        self.endX,self.endY = (event.x,event.y)
        self.imageforshow.coords(self.rect, self.startX, self.startY, self.endX, self.endY)

    # 选择结束，开始识别
    def delete_rect(self, event):
        # 消除矩形框
        self.imageforshow.delete(self.rect)
        # 根据矩形框的数据从当前图像中截取一个矩阵
        if self.startX > self.endX:
            self.endX, self.startX = self.startX, self.endX
        if self.startY > self.endY:
            self.endY, self.startY = self.startY, self.endY
        im = self.im_src[self.startY:self.endY, self.startX:self.endX]
        # 将圈出来的感受野放在右侧图像
        self.show_image(self.pre_process(im), position='right')
        # 对这个图像进行识别
        self.recognize(im)

    # 打开图片的点击响应
    def open_im_dir(self):
        files = [("PNG图片", "*.png"), ("JPG(JPEG)图片", "*.j[e]{0,1}pg"), ("所有文件", "*")]
        self.file_path_open = filedialog.askopenfilename(title="选择图片", filetypes=files)
        if len(self.file_path_open) is not 0:
            self.im_src = window.cv_imread(self.file_path_open)
            self.show_image(self.im_src,position='left')
            # 将图像放在右侧区域
            pass

    # 打开模型文件的点击响应
    def open_model_dir(self):
        files = [("Keras模型",".h5"),("tensorflow1.0模型",".pkl"),("所有文件", "*")]
        self.model_path_open = filedialog.askopenfilename(title="选择模型文件, 取消则使用默认模型", filetypes=files)
        if self.model_path_open != "":
            self.model_type = "keras"
            self.show_text.delete('1.0',END)
            self.show_text.insert(INSERT,"模型已导入，正在使用模型%s"%(self.model_path_open))
            self.model = models.load_model(self.model_path_open)

        else:
            self.model_type = "default"
            self.model = joblib.load('model/模型/BP.pkl')  # 读取模型
            self.show_text.delete('1.0',END)
            self.show_text.insert(INSERT,"正在使用标准模型")
        self.cursor_index += 1
        self.root.update()

# 工具方法
    # 识别图像前的预处理
    def pre_process(self, im):
        if len(im.shape) == 3:
            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        _, im = cv.threshold(im, 0, 255, cv.THRESH_OTSU)
        im = cv.resize(im, (28, 28))
        # 保证图像为黑底白字的图像
        if len(im[im == 0]) < im.shape[0]*im.shape[1]:
            im = 255 - im
        # 对图像进行开和闭yuns
        im = cv.morphologyEx(im, cv.MORPH_OPEN, (2, 2))
        im = cv.morphologyEx(im, cv.MORPH_CLOSE, (2, 2))

        if self.model_type == "default":
            pass
        if self.model_type == "keras":
            im = im.reshape((1,28,28,1))

        return im

    # 识别图片的方法，把识别的结果显示在文本框中
    def recognize(self, im):
        self.predict_text.delete('1.0')
        if self.model_type == "default":
            # 将图像阈值化后拉伸为一维向量
            im = self.pre_process(im)
            im_flatten = im.reshape(im.shape[0] * im.shape[1])
            im_flatten = im_flatten / 255
            predict = self.model.test_png(im_flatten)
            self.predict_text.insert('%d.0'%(self.cursor_index+1), str(predict))
        if self.model_type == "keras":
            im = self.pre_process(im)
            predict = self.model.predict(im)
            predict = np.argmax(predict)
            self.predict_text.insert('%d.0'%(self.cursor_index+1), str(predict))

    # 显示图片
    def show_image(self,im,newsize:tuple=None,position = 'left'):
        if im is None:
            self.imshow = np.zeros(0)
        elif len(im.shape) == 3 or len(im.shape) == 2:
            self.im_show = im.copy()
        else:
            im = im.reshape((28,28))
            self.im_show = im.copy()
        if newsize is None:
            (imsizecol,imsizerow) = self.get_reshape_size(self.im_show)
        else:
            # 设置传入数组指定的窗口大小
            (imsizecol,imsizerow) = newsize
        self.im_show = cv.resize(self.im_show, (imsizecol, imsizerow))
        self.root.geometry("%sx%s"%(int(self.width_max*4/5),int(self.height_max*4/5)))

        # 如果需要将图像放在左侧区域
        if position == 'left':
            self.im_tk_left = window.im_np2im_tk(self.im_show)
            self.imageforshow.config(height = imsizerow,width=imsizecol)
            self.imageforshow.create_image(0,0,anchor=NW,image=self.im_tk_left)
        # 如果需要将图像放在右侧区域
        else:
            self.im_show = cv.resize(self.im_show, (200,200))
            self.im_tk_right = window.im_np2im_tk(self.im_show)
            self.reception_field.create_image(0,0,anchor=NW,image=self.im_tk_right)

        # 只能改一次源图像大小，为了在选择图片时正确框选出图片的对应区域
        if self.src_size_changed:
            self.im_src =\
                cv.resize(self.im_src, (imsizecol,imsizerow))
        self.src_size_changed = False
        self.root.update()

    # 获取能够最好地显示的图片大小
    def get_reshape_size(self,im):
        if len(im.shape) is 3:
            imsizerow, imsizecol, _ = im.shape
        elif len(im.shape) is 2:
            imsizerow, imsizecol = im.shape
        else:
            _,imsizerow,imsizecol,_ = im.shape

        #将图片调整至最佳大小
        if imsizecol > self.width_max*0.7 or imsizerow > self.height_max*0.7:
            if imsizecol > imsizerow:
                imsizerow = int(self.width_max*0.7*imsizerow/imsizecol)
                imsizecol = int(self.width_max*0.7)

            if imsizecol <= imsizerow:
                imsizecol = int(self.height_max*0.7*imsizecol/imsizerow)
                imsizerow = int(self.height_max*0.7)

        return (imsizecol,imsizerow)

    def im_np2im_tk(im):
        # 改变三通道排列顺序并将图像转换为可显示的类型
        if len(im.shape) == 3:
            im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        img = Image.fromarray(im)
        imTk = ImageTk.PhotoImage(image=img)
        return imTk

    def cv_imread(file_path=""):
        #编码格式转换
        cv_img = cv.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
        return cv_img

w = window()
