import pandas as pd
df = pd.read_csv('submission.csv')
df['is_duplicate'] = df['is_duplicate'].apply(lambda x: 0 if x == 1 else 1 )
del df['test_id']
df.to_csv('flippedValues.csv', sep=',')
