import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
class Util:
    def report_classification(model,df_train,df_test,X_features,y_feature):
        eğitme_sınıfı=np.unique(df_train[y_feature].values).tolist()
        sınama_sınıfı=np.unique(df_test[y_feature].values).tolist()
        assert(eğitme_sınıfı==sınama_sınıfı)
        sınıflar=eğitme_sınıfı
        X_eğitme=df_train[X_features].values.tolist()
        X_sınama=df_test[X_features].values.tolist()
        y_eğitme=df_train[y_feature].values.tolist()
        y_sınama=df_test[y_feature].values.tolist()
        y_eğitme_tahmin=model.predict(X_eğitme)
        y_sınama_tahmin=model.predict(X_sınama)
        report_cm(y_eğitme,y_sınama,y_eğitme_tahmin,sınıflar)
    def report_cm(y_eğitme,y_sınama,y_eğitme_tahmin,sınıflar):
        figürler,eksenler=plt.subplots(1,2,figsize=(10,5))
        cm_sınama=confusion_matrix(y_sınama,y_sınama_tahmin)
        df_cm_sınama=pd.DataFrame(cm_sınama,index=sınıflar,columns=sınıflar)
        ax=sns.heatmap(df_cm_sınama,annot=True,ax=eksenler[0],square=True)
        ax.set_title('Sınama CM')
        cm_eğitme=confusion_matrix(y_sınama,y_sınama_tahmini)
        df_cm_eğitme=pd.DataFrame(cm_eğitme,index=sınıflar,columns=sınıflar)
        print('-' * 20 + 'Sınama Performansı' + '-' * 20)
        print(classification_report(y_eğitme,y_eğitme_tahmin,target_names=sınıflar))
        print('doğruluk:',metrics.accuracy_score(y_sınama,y_sınama_tahmin))
        print('-' * 20 + 'Eğitme Performansı' + '-' * 20)
        print(classification_report(y_sınama,y_sınama_tahmin,target_names=sınıflar))
        print('doğruluk:',metrics.accuracy_score(y_eğitme,y_eğitme_tahmin,target_names=sınıflar))
import matplotlib.pyplot as plt
import pandas as pd
import pyspark
sc=pyspark.SparkContext('local[*]')
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
X1,Y1=make_classification(n_features=10,n_redundant=0,n_informative=1,n_clusters_per_class=1)
df=pd.DataFrame(X1,columns=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'])
df['label']=Y1
df.to_csv('gen.csv',index=False)
from pypspark.sql import SparkSession
spark=SparkSession \
    .builder \
    .appName("TestApp") \
    .config("spark.some.config.option","some-value") \
    .getOrCreate()
df=spark.read.format('com.databricks.spark.csv').options(header='true',inferschema='true').load('gen.csv')
df.cache()
df.createOrReplaceTempWiew("gen")
df.printSchema()
display(df.show(10))
display(df.columns)
display(df.dtypes)
df.samples(False,0.3).count()
sqlDF=spark.sql("SELECT c1,label FROM gen")
sqlDF.show(10)
c1=list(map(lambda r : r['c1'],sqlDF.collect()))
etiket=list(map(lambda r : r['label'],sqlDF.collect()))
list(zip(c1,etiket))
df.select("c1").show(10)
df.select(['select']).show(10)
df.filter(df['c1']>0).select(['c1','c2']).show(10)
sqlDF.toPandas().head(10)
from pypspark.ml.feature import VectorAssembler
from pypspark.ml.feature import Pipeline
from pyspark.ml.tuning import CrossValidator,TrainValidationSplit,ParamGridBuilder
from pypspark.ml.classification import RandomForestClassifier
rf=RandomForestClassifier(labelCol="label",featuresCol="features")
from pypspark.ml.evaluation  import MultiClassClassificationEvalator
evaluator=MulticlassClassificationEvaluator()
from pyspark.ml import Pipeline
pipeline=Pipeline(stages=[vecAssembler,pca,rf])
from pypspark.ml.feature import PCA
pca=PCA(k=5,inpurCol="va",outputCol="Features")
from pypspark.ml.tuning import CrossValidator,TrainValidationSplit,ParamGridBuilder
paramGrid=(ParamGridBuilder()
            .addGrid(rf.maxDepth,[2,4,6])
            .addGrid(rf.maxBins,[20,60])
            .addGrid(rf.numTrees,[5,20]
            .addGrid(rf.impurity,['gini','entropy'])
            .build()))
def get_validation(by='cv'):
    if by is 'cv':
        return CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=10
                              )
    elif by is 'tvs':
        return TrainValidationSplit(estimator=pipeline,
                                    estimatorParamMaps=paramGrid,
                                    evaluator=evaluator,
                                    trainratio=0.8)
    else:
        print("lütfen tvsden ve cvden birini seçiniz")
        return None

def evaluate(model,df_train,df_test):
    train_pred=model.transform(df_train)
    train_acc=evaluator.evaluate(train_pred)
    test_pred=model.transform(df_test)
    test_acc=evaluator.evaluate(test_pred)
    print('eğitme_doğruluğu:{},sınama_doğruluğu:{}'.format(train_acc,))
    return train_pred,test_pred
val=get_validation('cv')
model_cv=val.fit(df_train)
bestModel=model_cv.bestModel
finalPredictions=bestModel.transform(df_train)
evaluator.evaluate(finalPredictions)
train_pred,test_pred=evaluate(model_cv,df_train,df_test)
y_train=train_pred.select('label').toPandas().apply(lambda x:x[0],1).values.tolist()
y_train_pred=train_pred.select('prediction').toPandas().apply(lambda x:x[0],1).values.tolist()
y_test=test_pred.select('label').toPandas().apply(lambda x:x[0],1).values.tolist()
y_train_test=test_pred.select('prediction').toPandas().apply(lambda x:x[0],1).values.tolist()
Util.report_cm(y_eğitme,y_sınama,y_train_pred,y_train_test,['0','1'])
selected=train_pred.select("label","prediction","probability")
display(selected.show[10])


       
