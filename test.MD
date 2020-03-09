---
title: "R introduction to machine learning"
subtitle: '기계학습 소개-SEHnR, basic step 1' 
output:
  html_document: 
    toc: true
    toc_depth: 2
    toc_float: true
    number_sections: true
   #progressive: true
    #allow_skip: true
#runtime: shiny_prerendered
    
---
# 명명
1) outcome : 예측하고 싶은 것
2) features: 예측에 사용되는 것


outcome|feature~1~|feature~2~|...|feature~n~|
|:---:|:---:|:---:|:---:|:---:|
|Y~1,1~ |Y~1,2~|Y~1,3~|...|Y~1,n~|
|Y~2,1~ |Y~2,2~|Y~2,3~|...|Y~2,n~|
|:|:|:|...|:|
|Y~i,1~ |Y~i,2~|Y~i,3~|...|Y~i,n~|


```{r, message=FALSE }
library(dplyr)
library(knitr)
library(kableExtra)
library(dslabs)
library(caret)
data(heights)
```

Outcome (y)과 Predictors(feature) (x)를 정의해 보자
```{r}
y<-heights$sex     # outcome
x<-heights$height  # feature
```

# Training and Test set


```{r, message=FALSE, warning=FALSE}
set.seed(2007, sample.kind = 'Rounding')
library(caret)
test_index <-createDataPartition(y, times=1, p=0.5, list=FALSE)
```
`set.seed`는 무작위 추출시 값을 고정한 것이고, `times`는 얼마나 많은 표본을 `p`의 확률로 뽑을 것인가이다. 

```{r}
test_set <- heights[+test_index, ]
train_set <-heights[-test_index,]
```
로 test_set과 train_set을 구분한다. 
여기서 **train_set** 만을 이용하여 예측 모데릉ㄹ 만들고, **test_set**에서 예측정도를 확인할 것이다.

# Overall accuracy
정확도 검사를 위해 우선 결과를 단순 무작위로 예측해보자
즉,
```{r}
y_hat <-sample(c("Male", "Female"), length(test_index), replace=TRUE) %>% factor(levels=levels(test_set$sex))
```
`sample`명령으로 남/녀 중에 test_index 갯수 만큰, `replace=TRUE` 복원 추출을 하였다. 이후 `%>%`로 후속 조치를 하였는데, 남/녀의 수준을 test_set$sex의 수준 값과 동일하게 맞추었다(이래야 비교할 때 오류가 않남)

전체적 정확도는
```{r}
mean(y_hat==test_set$sex)
```
당연히 50%정도의 정확도를 갖는다. 

남자와 여자의 키를 구별해보자
```{r}
heights %>% group_by(sex) %>% summarize(mean(height), sd(height)) %>%kable()%>%kable_styling(full_width = F)
```
62인치 보다 큰 사람을 남자로 하여 예측해보자.
```{r message=FALSE}
library(purrr)
```

```{r}
cutoff <- seq(61, 70)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train_set$height > x, "Male", "Female") %>% 
    factor(levels = levels(test_set$sex))
  mean(y_hat == train_set$sex)
})
```
```{r}
qplot(cutoff, accuracy)
```

민감도 특이도도 조사해보 쉽게 하는 방법은 `confusionMatrix`를 사용하는 것이다.
```{r}
y_hat <- ifelse(test_set$height > 64, "Male", "Female") %>% 
  factor(levels = levels(test_set$sex))
y_hat <- factor(y_hat)
```
여기서 y_hat에height가 64보다 클때, 남자, 아니면 여자로 값을 준후, 아래의 confusionMatrix에서 test_set의 sex 값과 일치되는지 여부를 보는 것이다.
```{r}
cm <-confusionMatrix(data=y_hat, reference=test_set$sex)
```
2*2 표로 나타내 보면
```{r}
cm$table
```
```{r, echo=FALSE}
gg<-data.frame('Actually Positive'=c('True positives (TP)', 'False negatives (FN)'), 
                'Actually Negative'=c('False positives (FP)','True negatives (TN)'))
rownames(gg) <- c('Predicted positive', 'Predicted negative')
```
```{r, echo=FALSE }
kable(gg) %>%
  kable_styling(bootstrap_options = c("striped", "bordered"), full_width = F) #'hover'
```

| |Actually Positive|	Actually Negative |
|---|---|---|
|Predicted positive|True positives (TP)|False positives (FP)|
|Predicted negative|False negatives (FN)|True negatives (TN)|

기본 산출 값들을 나열해 보면 아래와 같다.
```{r}
cm$overall
```
몇가지 용어를 정리해 보면

|Measure of|Nmae1|Name2|Definition|Probability representation|
|---|---|---|---|---|---|
|sensitivity|TPR|Recall| $$ \frac{\mbox{TP}}{\mbox{TP} + \mbox{FN}} $$|$$ \mbox{Pr}(\hat{Y}=1 \mid Y=1)$$|
|specificity|TNR|1-FPR| $$\frac{\mbox{TN}}{\mbox{TN}+\mbox{FP}}$$|$$\mbox{Pr}(\hat{Y}=0 \mid Y=0)$$|
|specificity|PPV|Precision|$$\mbox{Pr}(\hat{Y}=0 \mid Y=0)$$|$$\mbox{Pr}(\hat{Y}=0 \mid Y=0)$$|

# F~1~ score and Balanced Accuracy
Specificity 와 Sensitivity 의 평균적 요약값이 F~1~ score이다. 
F~1~ score는 아래와 같이 표시할 수 있다.
$$ \frac{1}{\frac{1}{2}\left(\frac{1}{\mbox{recall(sensitivity)}} + 
    \frac{1}{\mbox{precision}}\right) } $$
`caret` package를 이용하면 쉽게 구할 수 있다.
```{r caret package}
library(caret)
F_meas(data=y_hat, reference=test_set$sex)
```
cutoff를 바꿔가면서 F_1을 구해보면 'cutoff=65'가 결정된다.
```{r}
F_1 <-map_dbl(cutoff, function(x){
  y_hat <-ifelse(train_set$height >x, "Male", "Female") %>% factor(levels=levels(test_set$sex))
  F_meas(data=y_hat, reference=factor(train_set$sex))
})
qplot(cutoff, F_1)+geom_line()
```

# 연속변소에서 Loss function
지금가지 2분형 변수에 대해서 민간도, 특이도, 정확도, F_1을 연습했다. 그러나 outcome이 연속변수 일경우에는 `loss function`을 사용할 것이다.
가장 많이 사용되는 `loss fucntion`은 
$\hat{y}$ 와 $y$ 값의 차이를 줄여주는 방법을 사용할 수 있다.
$$(\hat{y}-y)^2$$
여러 값이 존재하므로 평균값을 구하면,
$$\mbox{MSE} = \frac{1}{N} \mbox{RSS} = \frac{1}{N}\sum_{i=1}^N (\hat{y}_i - y_i)^2$$
실제적으로는 `root mean squared error (RMSE)`를 흔히 사용하며, $\sqrt{\mbox{MSE}}$ 로 계산한다. RMSE가 줄어드는 방향으로 모델을 개발해 가는 것이다.
