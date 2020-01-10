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
