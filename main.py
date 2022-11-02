from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv', sep=",")

plt.title('Experience Vs. Salary')
plt.xlabel('experience(#ofyears)')
plt.ylabel('salary(US$)')
plt.scatter(data.YearsExperience, data.Salary, color="red", marker='+')

# plt.show()

reg = LinearRegression()
reg.fit(data[['YearsExperience']].values, data['Salary'])


valueToPredict = int(input("Years of Experience: "))
print(reg.predict([[valueToPredict]]))