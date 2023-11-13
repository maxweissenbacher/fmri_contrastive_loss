import pingouin
import pandas
from collections import Counter
import numpy as np


def icc_full(subjects, values, version="ICC1"):
    counts = Counter(subjects)
    assert len(set(counts.values())) == 1, "Different numbers of subject ratings in ICC"
    df = pandas.DataFrame({"subject": subjects, "value": values})
    df.sort_values("subject", inplace=True, kind="mergesort")  # mergesort is only stable sort
    df['rater'] = np.tile(range(0, len(subjects) // len(set(subjects))), len(set(subjects)))
    iccs = pingouin.intraclass_corr(data=df, targets="subject", raters="rater", ratings="value")
    iccs.set_index('Type', inplace=True)
    return iccs.loc[version]['ICC'], tuple(iccs.loc[version]['CI95%']), iccs.loc[version]['pval']
