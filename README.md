# DeepWater
~~~
水环境数据智能预报功能
~~~


## 运行环境
~~~
本项目基于tensorflow2.0开发深度学习相关部分，由于项目需要深度学习相关库，部门服务器无法满足环境需求；
另配置miniconda3环境，相关环境打包文件可联系[buzh@3clear.com]，感谢关注和支持。
~~~

+ 将miniconda3打包环境解压到指定位置，修改```scripts/run.sh```中```export PYTHON="{your_abspath}/miniconda3/bin/python"```


## config
~~~
项目总体配置目录，包含公共配置和模块配置(借鉴学习[lava]架构👍)
~~~
+ ```common.py``` 公共配置模块，设计用于整体框架相关配置

```
主要配置信息：

1.  proj_home       项目路径

2.  common_params   公共配置内容
|__ FREQ            数据频率
|__ SQLite          数据库路径
|__ PostgreSQL      数据库信息

3.  forecast_params 预报配置内容
|__ 预报公共配置
|__ Arima相关配置
|__ Fbprophet相关配置
|__ LSTM相关配置
```


## dao 
~~~
DAO(Data Access Object)数据对接模块，设计功能包括数据库、文本数据对接，其中包含实时数据、离线数据对接。
~~~
+ ```settings.py``` 模块内测试配置文件，后期统一到config模块内

+ ```test.py``` 模块内测试脚本，测试数据读写功能

+ orm 关系映射   

```db_orm.py``` 数据库连接    

```SQLite_Static.py``` 手动更新数据表模型 **不推荐**    

```SQLite_dynamic.py``` 自动获取数据库对应表数据模型     

```PostgreSQL_dynamic.py``` 对接业务流程PostgreSQL的数据模型 



## data
~~~
开发、测试数据目录，建议使用数据库。
~~~

+ ```waterTestDB.db``` 本地数据信息化及开发用的SQLite数据库



## docs
~~~
文档目录
~~~
+ 统计预报说明文档.pdf
+ ARIMA模型.pdf
+ fbprophet_paper.pdf



## lib
~~~
算法库封装模块
~~~

#### 机器学习算法

+ ```LinearModel```    线性模型

+ ```NonLinearModel``` 非线性模型

+ ```EnsembleModel```  集合模型

#### 深度学习算法

  **时间序列的神经网络模型**

+ ```SingleShot``` 并行多指标一次性预测多时次框架

+ ```DataWindow``` 滑动数据集构建器 

#### 统计学习算法

+ ```ARIMA```       Arima算法

+ ```fbprophet```   fbprophet算法



## logs
~~~
日志输出目录，其中 *.log为设计日志记录功能输出，*.out为当次任务打印输出信息。
~~~



## notebooks
~~~
包含数据前分析阶段的notebook，可在jupyter-notebook & jupyter-lab & VSCode等环境中打开；
EDA(Exploratory Data Analysis)，数据探索性分析是指在不清楚数据内容、质量、分布的情况下，对数据进行不同纬度的探索性分析；
EDA的目标是前期不受行业经验限制，对数据 进行统计分析，之后结合专业经验提升对于数据的感知能力。
~~~
+ **.ipynb文件环境与运行环境可能存在一些库版本不同**



## scripts
~~~
控制任务脚本。
~~~

+ ```run_forecast.py``` 运行不同模型脚本
+ ```run.sh```          运行项目主脚本，需提供预测时间，时间格式为{$yyyy$mm$dd$hh}
+ ```task2file.py```    用于生成项目并行运行的任务列表文件脚本



## src
~~~
任务 & 功能模块
~~~

#### DataSimulation
```
基于五项常规监测模拟四项水质指标的数据处理&非线性模型训练、预测任务。

功能设计：
基于五项常规监测指标{水温、PH、电导率、浊度、溶解氧}，建立线性&非线性模型模拟四项水质指标{高锰酸盐、氨氮、总磷、总氮}数据；
进而与水质指标线性插值填补数据进行对比，弥补间隔时间过长的线性填补劣势。
```


#### DataForecast ✅
```
四项水质指标预测功能任务，包括ARIMA模型、机器学习模型、深度学习模型。
```


#### DataQC ✅
```
对接总站水质监测数据的质控处理模块，包括：
1. 统计上下限分位数剔除高低异常值;
2. 线性插值填补;
3. 前向或后向填补;
```
+ 运行方式

```
cd {$your_proj_home}/DeepWater/scripts;
{$PYTHON} -n '站点名称' -m 'qc' -i 'all' -s '2020060600' -e '2020101000' 
```
```
-n	站点名称
-m	模型名称，此处唯一为qc
-i	质控指标，此处唯一为all，代表五常+四项
-s	质控时段开始时间，一般小时位为00、04、08、12、16、20
-e	质控时段结束时间，一般小时位为00、04、08、12、16、20
```


#### DataDiagnose
```
数据诊断功能
```

+ TODO

#### DataEvaluate
```
预测评估功能
```

+ TODO



## task
~~~
任务记录目录
~~~

+ 名称为{task_yyyymmddHH.txt}的文件记录了该预报时次所有任务，方便pbatch调用进行并行计算。

文件样例
```
cd /public/home/buzh/water/DeepWater/scripts; /public/home/buzh/env/miniconda3/bin/python run_forecast.py -n '白马寺' -m 'Arima' -i 'tn' -s '2020051108' -e '2020051108' >& /public/home/buzh/water/DeepWater/logs/logs_forecast/task_2020051108.out
cd /public/home/buzh/water/DeepWater/scripts; /public/home/buzh/env/miniconda3/bin/python run_forecast.py -n '白马寺' -m 'Fbprophet' -i 'tn' -s '2020051108' -e '2020051108' >& /public/home/buzh/water/DeepWater/logs/logs_forecast/task_2020051108.out
cd /public/home/buzh/water/DeepWater/scripts; /public/home/buzh/env/miniconda3/bin/python run_forecast.py -n '白马寺' -m 'LSTM' -i 'tn' -s '2020051108' -e '2020051108' >& /public/home/buzh/water/DeepWater/logs/logs_forecast/task_2020051108.out
```



## utils
~~~
整体框架工具模块
~~~

+ ```ConfigParseUtils.py``` 命令行解析工具🔧

+ ```ConstantUtils.py```    常量枚举类工具🔧

+ ```FileUtils.py```        文本文件交互工具🔧 

+ ```LogUtils.py```         日志工具🔧

+ ```SimulateUtils.py```    DataSimulation模块专用工具🔧

+ ```ThreadUtils.py```      多线程类工具🔧



## host_file
~~~
多节点核心数配置文件
~~~

+ 利用mpirun运行时于-f指定使用。

+ 修改```scripts/run.sh```中```MPI_HOST```路径



## pbatch
~~~
批量运行任务文件程序
~~~

+ pbatch和host_file的使用均在scripts中run.sh脚本内。

+ 修改```scripts/run.sh```中```PBATCH```路径



## 运行方式

### scripts/run.sh

+ 解压运行环境miniconda3，其路径修改在```run.sh```中的```export PYTHON="{your_abspath}/miniconda3/bin/python"```

+ ```git clone ssh://git@47.92.132.84:2000/buzh/DeepWater.git```

+ ```git checkout dev```分支（目前dev为最新运行版）

+ 修改```run.sh```中```export PROJ_HOME={your_abspath}/DeepWater```

+ 修改```run.sh```中```export MPIRUN={your_abspath}```

+ 修改```host_file```中节点及核心数信息，根据节点资源和测试方式选择```run.sh```中mpirun的方式

### config/common.py

+ 修改```proj_home```
+ 数据库信息```SQLite_dir``` 、```PostgreSQL_info```
+ 预报配置参数：预报指标列表```INDICES``` 、预报模型列表```MODELS``` 、 预报站点列表```STATIONS```
+ 预报算法配置：历史拟合天数```last_days``` 、 输入输出序列长度```IN&OUT_STEPS```
