import pandas as pd


def pad_truncate(df, n):
    groups = df.groupby([df.local_time.dt.date, df.local_time.dt.hour])
    gs = []
    i = 0
    for label, group in groups:
        group = group.head(n)
        missing = n - len(group)
        new_rows = []
        cols = list(group.columns)
        for _ in range(missing):
            new_rows.append({col: 0 for col in cols})
            i += 1
        gs.append(pd.concat([group, pd.DataFrame(new_rows)]))
    return gs, i
