(task1)=

# Task 1
<!-- hack to open links in new tab
<head>
    <base target="_blank">
</head> -->

(task1:chooseatopic)=

## Choosing a Topic

The approval form ensures you start in the right direction before investing time and effort into task 2. Evaluators look for our ([instructors'](ci_page)) signature, and we look for the following:

1. A research question or 'organizational need.'
2. A statistical test or model supporting the research question or organizational need.
3. A data source or plan to obtain data.

```{margin}
The capstone can be very similar to a combination of tasks 1 and 2 from [C749 Intro to Data Analysis & Practical Statistics](https://learn.udacity.com/nanodegrees/nd002-wgu-1); *Quiz 5: Hypothesis testing* shows examples of a two-sample z-test and regression, either which can satisfy part 2 above.  
```

Task 1 is a preliminary exercise checking that your topic has the essential elements of a passing project. It is *not* an exact blueprint to which you'll be held accountable. After investing time and effort, deviating from details on the topic approval form is common and allowed. However, as deviating significantly from what was approved could put you at risk of not meeting the requirements, substantial changes should be discussed with your assigned course instructor.

<!-- NEED VIDEO
- **Watch:** Choosing a Project Idea
<iframe
    src="https://wgu.hosted.panopto.com/Panopto/Pages/Embed.aspx?id=691a0e9a-d33b-48ca-aac3-d7413f4bbfdc&autoplay=false&offerviewer=true&showtitle=true&showbrand=true&captions=true&interactivity=all" 
    title="Choosing a topic" 
    width="640px"
    height="360px"
    style="border: 1px solid #464646;"
    allowfullscreen allow="autoplay"
>
</iframe> -->

<!-- NEED PODCAST
- **Listen:**
IT Audio Series podcast [Choosing Your Topic](https://d2y36twrtb17ty.cloudfront.net/sessions/09c3d32b-1567-4636-88d6-ad5f01616619/3183eabf-4239-4e4a-ae20-ad5f01616620-8792ef88-6d24-43f6-9fe9-ad5f01619cc7.mp4?invocationId=0275eb54-33e0-eb11-8284-12c206d2fd2b); view the [transcript](https://www.wgu.edu/content/dam/western-governors/documents/it/audio-series/ChoosingYourTopic.docx). -->

To get an understanding of what's typically expected, review these [tasks 1-3 examples](resources:examples). Examples can also be found in the [Capstone Excellence Archive](https://westerngovernorsuniversity.sharepoint.com/sites/capstonearchives/excellence/Pages/UndergraduateInformation.aspx) which includes a wide range of completed projects. However, keep in mind that they all are, by definition, *above and beyond* the requirements. Therefore, do not use these to set expectations of what's needed to fulfill the requirements. For a more down-to-earth example of what's required, [tasks 1-3 examples](resources:examples).

### Data

You can’t perform data analysis without data. You will need to find and choose your own data. Any open source data set is freely available for use. The IRB policy only applies to data collected by you.

- [Kaggle.com](https://www.kaggle.com/datasets)
- [OpenML](https://openml.org/search?type=data&sort=runs&status=active). [Importing OpenML data](https://www.openml.org/apis)
- [Google Dataset Search](https://datasetsearch.research.google.com/)
- [Data.gov](https://data.gov/)
- Simulated data
- Python library built-in datasets:  
  - [sklearn's data sets](https://scikit-learn.org/stable/datasets.html) (these can be imported directly into your code)
  - PyTorch's built-in datasets: [images](https://pytorch.org/vision/stable/datasets.html), [texts](https://pytorch.org/text/stable/datasets.html), and [audio](https://pytorch.org/audio/stable/datasets.html)
  - [Tensorflow datasets](https://www.tensorflow.org/datasets)
- More [here](https://careerfoundry.com/en/blog/data-analytics/where-to-find-free-datasets/) and [here](https://medium.com/analytics-vidhya/top-100-open-source-datasets-for-data-science-cd5a8d67cc3d)

```{note}
*No minimal data complexity or processing is required.* Choosing data which needs less processing or simplifying a dataset (you don't have to use it al; indeed, sometimes you shouldn't) can make the project technically more accessible.  
```

The only explicit requirement regarding your chosen data is that it be available to use and share with evaluators. For almost any data set, a suitable research question or organizational need can be found. However, it will be best that the supporting statistical test or model fits both your interests and existing skill set. Therefore, when browsing through data sets, you might do so through the perspective of finding a data set well suited for your preferred method(s).  

(task1:topicapproval)=

### Statistical test or Model

From the task 2 rubric:
> **C4:METHODS AND METRICS TO EVALUATE STATISTICAL SIGNIFICANCE**
> *The submission thoroughly and accurately describes the methods and metrics. The description includes specific details on how the <mark style="background-color:yellow">methods and metrics will evaluate **statistical significance**.*</mark>*

From the task 3 rubric: 
> **F1:STATISTICAL SIGNIFICANCE**
> *A thorough evaluation of the <mark style="background-color:yellow">statistical significance of the analysis is provided</mark>, and the evaluation uses accurate calculations.*

You will need to make an argument supporting a hypothesis using appropriate data analytic methods. An inductive argument using only descriptive methods will not suffice. Ideas supportable with hypothesis testing, e.g., claims about correlations, means, proportions, etc., as using statistical significance best fits the requirements, but models can also be used.

#### Statistical Significance

Statistically significant results can be found by applying an appropriate inferential statistical test. Examples include:

- Correlation tests, e.g., a hypothesis test showing two variables are correlated.
- Comparison tests, e.g., a hypothesis test showing two means are different.
- Estimating parameters, e.g., a confidence interval estimating a mean.

Inferential statistical methods and testing for statistical significance are covered in [C749 Intro to Data Analysis & Practical Statistics](https://learn.udacity.com/nanodegrees/nd002-wgu-1). For a brief overview, listen to [C964/D502 Statistical Significance](https://d2y36twrtb17ty.cloudfront.net/sessions/d8c9630b-274a-4879-bf27-ae2a011e3dbd/ec782192-a9bf-459f-b091-ae2a011e3dc8-8a3dfb1b-4294-4074-9d33-ae2a011e752c.mp4?invocationId=abd666da-b37f-ec11-828b-12b1cb861383) in the [WGU IT Audio series](https://www.wgu.edu/online-it-degrees/it-audio-series.html#close).
<!-- Add a stat sign video -->

#### Models

While the rubric explicitly requires "statistical significance," not all non-descriptive rigorous data analysis methods have a known probability space from which to derive a $p-\text{value}$, e.g., many common machine learning models; but using such (statistical or machine learning) models can satisfy the "statistical significance" requirements of tasks 2 and 3.
<!-- OK there are some things here to get a p-value but they are sus and not highly developed; see that paper I downloaded... -->

:::{margin} ML or Statistical Model?
Machine learning (ML) is a new and fast-developing field. The overlap of methods, algorithms, language, etc., makes the two at times indistinguishable. But they are different. Statistics is a mathematical science while ML is an approach to solving problems that would otherwise be cost-prohibitive, e.g., very large datasets. Similar to how calculus is related to engineering, statistics is one of many tools used by ML, and like engineering ML focuses on results and application. For this project, statistical and ML methods are both acceptable, but you need not distinguish between the two.
:::

Examples include:

- Supervised regression models with performance measured by an appropriate metric such as [mean squared error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) or [coefficient of determination](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html#sklearn.metrics.r2_score).
- Supervised classification models with performance measured by an appropriate [accuracy score](https://scikit-learn.org/stable/modules/classes.html#classification-metrics)](https://scikit-learn.org/stable/modules/classes.html#classification-metrics).
- Unsupervised models such as [clustering](https://scikit-learn.org/stable/unsupervised_learning.html#unsupervised-learning) with performance measured using a [clustering metirc](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) such as the rand score or silhouette coefficient.

## Topic Approval

Once you've decided on a topic, complete the approval form following the template:
<!-- TODO: ADD THUMBNAIL IMAGE
<!-- > [![Task 1 example](https://github.com/ashejim/C964/blob/main/url_images/example_task1-a.png?raw=true#image-thumb)](https://github.com/ashejim/C964/blob/main/resources/example_task1-a.pdf) -->
<!-- TODO: CHECK THIS LINK -->

> [D502 Topic Approval Form](https://1drv.ms/w/s!Av4KQnJfiBxmhI0Q1vuAGcetB5UdFg?e=1gyubA)

Include a rough outline of your research question or organizational need, supporting inferential method or model, and implementation. **Email the completed form to** **[your course instructor](ci_page)** who will either approve it with their signature or provide feedback.

```{note}
The topic approval form must be *signed by a* *[D502 course instructor](ci_page)*. Forms without a signature are automatically returned without further review.  
```

Directly emailing your assigned course instructor is the fastest and often best way to get a signature. Whether emailing [ugcapstoneit@wgu.edu](mailto:ugcapstoneit@wgu.edu?cc=my%20course%20instructor&subject=C769:%20capstone%20topic%20approval&body=Your%20name%20and%20question%20here.%20We%20can%20only%20respond%20to%20messages%20from%20a%20valid%20WGU%20email%20address.%20%0A%0ADegree%20program%3A%20%0AProgram%20Mentor%3A%20%0A) or your CI directly, always practice professional communication:

- Use your WGU email.
- Provide a subject, your capstone course, and your program mentor's name (if not in your signature)
- Clearly state your questions or requests.

(task1:waiverform)=

## Waiver Form

<!-- Everyone must submit a waiver form stating either their project is not based on restricted information OR use of any restricted information is authorized. -->

```{note}
the waiver form is **only** required if your project includes restricted information. If no waiver form is submitted, Task 1 *B: Capstone Release Form*, passes automatically.
```

In most cases, obtaining authorization can be avoided by masking or removing identifying information. But if you choose to move forward using restricted information, you must obtain documented permissions and submit them along with a waiver form to Assessments.

<!-- > [![Waiver Form](https://github.com/ashejim/C769/blob/main/url_images/769_waiver_form_thumb-1.png?raw=true#image-thumb)](https://westerngovernorsuniversity-my.sharepoint.com/:w:/g/personal/jim_ashe_wgu_edu/EUNAmf7lWqxOmKBLWTQ_zPcByoxrOLLK5sILQeeUoeYGeQ?e=9d1Ef7) -->

> [D502 Waiver Form](https://1drv.ms/w/s!Av4KQnJfiBxmgsFqcRinq2h4XBjFrw?e=KHpX5q)

(task1:faq)=

## FAQ

### Do I need to set up an appointment to get approval?

No. Usually, students email the approval form to their instructor. We then sign to form or follow up with questions. However, if you have questions about the requirements or difficulty choosing a topic, you are encouraged to set up an appointment with [your course instructor](ci_page). A 15-30 minute phone call can address most questions or concerns. If you do set up an appointment to discuss your approval form, please email it to the instructor before the appointment.

### Are there any examples?

Yes! See [D502 examples](resources:examples).

### What if I start writing task 2 and want to change things? Do I need to resubmit task 1?

No. Minor changes from task 1 to task 2 are expected and allowed *without updating the approval form*. Evaluators will not rigorously compare tasks 1 and 2. Task 2 is where the work is, and even with complete topic changes at most, you might need to revise the approval form (if at all). So never let task 1 dictate what you do in task 2. However, deviating significantly from what was approved could put you at risk of completing a project not meeting the requirements. So while small changes do not need review, substantial changes should be discussed with your assigned course instructor.

### Do I need an "electronic signature" as specified in the official rubric?

You can type in your name, use a "fancy" font, or insert an image of your signature.  

### What are the common reasons for task 1 being returned?

1. No instructor signature on the approval form. You need to send it to us and get a signature *before* submitting it. Both boxes or no boxes are correctly marked on the waiver form. *Mark one and only one box*.
2. Both or neither box is marked on the waiver form. Mark one and only one box. See the [waiver form instructions](task1:waiver)

Note, the waiver form is **only** required if your project includes restricted information. Task 1 *B: Capstone Release Form*, passes automatically if no waiver form is submitted, i.e., the waiver is only needed if it's needed.

### How many attempts are allowed for each assessment?

You have unlimited attempts. However, incomplete submissions or submissions significantly falling short of the minimum requirements may be *locked* from further submissions without instructor approval. Furthermore, such submissions do not receive meaningful evaluator comments.

(task1:faq:hard)=
 
### Can I get the "welcome email?"

Yes, contact your assigned course instructor or see the sample [welcome email](resources:general:welcomeemail) in the [D502 Resources section](resources).

<!-- ### Are there any cohorts? I don't see where to sign-up on my COS page.

Yes, cohorts run regularly. Enrollment typically opens on Mondays and closes Wednesdays. You can find the link to sign-up under the *Explore Cohort* section on your C769 COS page. If the section is not visible, either enrollment has closed or the cohort will not be available that week. See [Webinars and Cohorts](resources:cohort) for more details. -->

### Can I use projects from other WGU courses?

Yes! You can use any of your work or academic projects (at WGU or elsewhere) provided no proprietary information is used without permission. Don't worry about self-plagiarism, as the similarity check will identify and ignore it. Just as in reusing work projects, expect to modify and remold past academic assignments to meet the rubric requirements.

### How complex does my data or analytic method need to be?

It must be complex enough to meet the needs of your project. There is no explicit minimal complexity for either. However, the data must meet the needs of the research question and the method must be appropriate for both the data and the research question which may indirectly require a minimal complexity. For example, testing for correlation inherently requires two variables and parametric methods often need a minimal sample number to assume normality.

### Are there any restrictions on which datasets I can choose?

Only that data must be legally available to use and share with evaluators. For example, using data belonging to a current employer would require submitting a [waiver form](task1:waiverform).

- You *can* use any dataset found on [kaggle.com](https://www.kaggle.com/datasets).
- You *can* use simulated data.
- You *can* use data used for previous projects (submitted by you or others).
- You only need to apply for [IRB review](https://cm.wgu.edu/t5/Frequently-Asked-Questions/WGU-IRB-and-Human-Subject-Protections-FAQ/ta-p/2002) if you are *collecting* data involving human participants (this is rarely needed). Otherwise, your project is in IRB compliance.

## Questions, comments, or suggestions?

<script
   type="text/javascript"
   src="https://utteranc.es/client.js"
   async="async"
   repo="ashejim/D195"
   issue-term="pathname"
   theme="github-light"
   label="💬 comment"
   crossorigin="anonymous"
/>