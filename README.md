# ML1

## Remarks

I've added more details in the notebook, mostly adding quick analysis to the data and trying other learning method (logistic regression, Cyril has already experimented with k-NN).

My ideas (and taken from other from *Discussion* forum as well):

 - We cannot use all features to train model, otherwise the results might not be accurate (refer to the portion of codes where I apply logistic regression to the original data and have ~ 98% of accuracy, which is...too good to be true).
 - The meaning of the features (e.g. *reg*, *car*, *calc*) is not easy to interpret.
 - Should we apply **dimensionality reduction** ? For example, if we know that features of the same grouppings share the same suffixes (*reg_x*, *car_y*), is there any way to "group" them together ?
 - Should **correlation** between data be taken into account ?
 - How to properly deal with features of **different natures**: should we treat numerical, ordinal, categorical features differently? If yes, then how?
 - I heard that all *calc_x* features do not contribute much to the data (at least that's what everyone is spreading wild in Discussion forum). So, **feature selection** is important, in order to refine the prediction. I've added a code that compute feature importance, and noticed that, indeed, all features of type *calc_x* contribute little to the output.