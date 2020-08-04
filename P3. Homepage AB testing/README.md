## Scenario Description

Let's say that you're working for a fictional productivity software company that is looking for ways to increase the number of people who pay for their software. The way that the software is currently set up, users can download and use the software free of charge, for a 7-day trial. After the end of the trial, users are required to pay for a license to continue using the software.

One idea that the company wants to try is to change the layout of the homepage to emphasize more prominently and higher up on the page that there is a 7-day trial available for the company's software. The current fear is that some potential users are missing out on using the software because of a lack of awareness of the trial period. If more people download the software and use it in the trial period, the hope is that this entices more people to make a purchase after seeing what the software can do.

In this case study, you'll go through steps for planning out an experiment to test the new homepage. You will start by constructing a user funnel and deciding on metrics to track. You'll also perform experiment sizing to see how long it should be run. Afterwards, you'll be given some data collected for the experiment, perform statistical tests to analyze the results, and come to conclusions regarding how effective the new homepage changes were for bringing in more users.

## Metrics 

### Invariant Metric
The number of cookies assigned to each group

### Evaluation Metrics
1. Download rate
2. License purchasing rate

## Conclusion

For the test of the invariant metric, number of cookies, there were a larger number of cookies recorded in the experiment group, 47 346 vs. 46 851. This ends up generating a p-value of 0.107 (z = -1.61), which is within a reasonable range under the null hypothesis. Since we lack sufficient reason to reject the null, we can continue on to evaluating the evaluation metrics. (Note that this doesn't mean that there wasn't something actually different about the cookie counts between groups, only that we couldn't detect it if such a difference existed.)

For the first evaluation metric, download rate, there was an extremely convincing effect. An absolute increase from 0.1612 to 0.1805 results in a z-score of 7.87, well beyond any standard significance bound. However, the second evaluation metric, license purchasing rate, only shows a small increase from 0.0210 to 0.0213 (following the assumption that only the first 21 days of cookies account for all purchases). This results in a p-value of 0.398 (z = 0.26).

Despite the fact that statistical significance wasn't obtained for the number of licenses purchased, the new homepage appeared to have a strong effect on the number of downloads made. Based on our goals, this seems enough to suggest replacing the old homepage with the new homepage. Establishing whether there was a significant increase in the number of license purchases, either through the rate or the increase in the number of homepage visits, will need to wait for further experiments or data collection.

One inference we might like to make is that the new homepage attracted new users who would not normally try out the program, but that these new users didn't convert to purchases at the same rate as the existing user base. This is a nice story to tell, but we can't actually say that with the data as given. In order to make this inference, we would need more detailed information about individual visitors that isn't available. However, if the software did have the capability of reporting usage statistics, that might be a way of seeing if certain profiles are more likely to purchase a license. This might then open additional ideas for improving revenue.