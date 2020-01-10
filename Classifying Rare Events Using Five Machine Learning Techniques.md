# Classifying Rare Events Using Five Machine Learning Techniques

## Background
A couple years ago, Harvard Business Review released an article with the following title [“Data Scientist: The Sexiest Job of the 21st Century.”](https://hbr.org/2012/10/data-scientist-the-sexiest-job-of-the-21st-century) Ever since its release, Data Science or Statistics Departments become widely pursued by college students and, and Data Scientists (Nerds), for the first time, is referred to as being sexy.

For some industries, Data Scientists have reshaped the corporation structure and reallocated a lot of decision-makings to the “front-line” workers. Being able to generate useful business insights from data has never been so easy.

According to Andrew Ng ([Machine Learning Yearning](https://www.deeplearning.ai/machine-learning-yearning/), p.9),
> "Supervised Learning algorithms contribute the majority value to the industry."

There is no doubt why SL generates so much business value. Banks use it to detect credit card fraud, traders make purchase decisions based on what models tell them to, and factory filter through the production line for defective units (this is an area where AI and ML can help traditional companies, according to Andrew Ng).

These business scenarios share two common features:
**Binary Results:** fraud VS not fraud, to buy VS not to buy, and defective VS not defective.
**Imbalanced Data Distribution:** one majority group VS one minority group.
As Andrew Ng points out recently, small data, robustness, and human factor are three obstacles to successful AI projects. To a certain degree, our rare event question with one minority group is also a small data question: the ML algorithm learns more from the majority group and may easily misclassify the small data group.

Here are the million-dollar questions:
> For these rare events, which ML method performs better?

> What metrics?

> Tradeoffs?

In this post, we try to answer these questions by applying 5 ML methods to a real-life dataset with comprehensive R implementations.

For the full description and the original dataset, please check the original [dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing); For the complete R code, please check [Leihua Ye's Github](https://github.com/LeihuaYe/Machine-Learning-Classification-for-Imbalanced-Data).

## Business Question
A bank in Portugal carries out a marketing strategy of a new banking service (a term deposit) and wants to know which types of clients have subscribed to the service. So, the bank can adjust its marketing strategy and target specific groups of populations in the future. Data Scientists have teamed up with the sells and marketing teams to come up with statistical solutions to identify future subscribers.

## R Implementations
Here comes the pipeline of model selection and R implementations.

### 1. Importation, Data Cleaning, and Exploratory Data Analysis
Let’s load and clean the raw dataset.

```
####load the dataset
banking=read.csv(“bank-additional-full.csv”,sep =”;”,header=T)

##check for missing data and make sure no missing data
banking[!complete.cases(banking),]

#re-code qualitative (factor) variables into numeric
banking$job= recode(banking$job, “‘admin.’=1;’blue-collar’=2;’entrepreneur’=3;’housemaid’=4;’management’=5;’retired’=6;’self-employed’=7;’services’=8;’student’=9;’technician’=10;’unemployed’=11;’unknown’=12”)

#recode variable again
banking$marital = recode(banking$marital, “‘divorced’=1;’married’=2;’single’=3;’unknown’=4”)

banking$education = recode(banking$education, “‘basic.4y’=1;’basic.6y’=2;’basic.9y’=3;’high.school’=4;’illiterate’=5;’professional.course’=6;’university.degree’=7;’unknown’=8”)

banking$default = recode(banking$default, “‘no’=1;’yes’=2;’unknown’=3”)

banking$housing = recode(banking$housing, “‘no’=1;’yes’=2;’unknown’=3”)

banking$loan = recode(banking$loan, “‘no’=1;’yes’=2;’unknown’=3”)
banking$contact = recode(banking$loan, “‘cellular’=1;’telephone’=2;”)

banking$month = recode(banking$month, “‘mar’=1;’apr’=2;’may’=3;’jun’=4;’jul’=5;’aug’=6;’sep’=7;’oct’=8;’nov’=9;’dec’=10”)

banking$day_of_week = recode(banking$day_of_week, “‘mon’=1;’tue’=2;’wed’=3;’thu’=4;’fri’=5;”)

banking$poutcome = recode(banking$poutcome, “‘failure’=1;’nonexistent’=2;’success’=3;”)

#remove variable “pdays”, b/c it has no variation
banking$pdays=NULL 

#remove variable “pdays”, b/c itis collinear with the DV
banking$duration=NULL
```

It appears to be tedious to clean the raw data as we have to recode missing variables and transform qualitative into quantitative variables. It takes even more time to clean the data in the real world. **There is a saying “data scientists spend 80% of their time cleaning data and 20% building a model.”**

Next, let’s explore the distribution of our outcome variables.

```
#EDA of the DV
plot(banking$y,main="Plot 1: Distribution of Dependent Variable")
```

