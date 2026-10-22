import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Apply Seaborn Theme
sns.set_theme(style="whitegrid")

# -------------------------------
# BAR CHART
# -------------------------------
students = ['A', 'B', 'C', 'D', 'E']
marks = [85, 92, 78, 88, 95]

plt.figure(figsize=(8,5))
bars = plt.bar(students, marks, color='skyblue')

plt.title("Student Marks")
plt.xlabel("Students")
plt.ylabel("Marks")

for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height()+1,
             str(bar.get_height()),
             ha='center')

plt.grid(axis='y')
plt.show()

# -------------------------------
# LINE GRAPH
# -------------------------------
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales = [120, 150, 180, 170, 200]

plt.figure(figsize=(8,5))
plt.plot(months, sales,
         marker='o',
         color='green',
         linewidth=2)

plt.title("Monthly Sales Trend")
plt.xlabel("Months")
plt.ylabel("Sales")

for i in range(len(months)):
    plt.annotate(sales[i],
                 (months[i], sales[i]),
                 xytext=(0,10),
                 textcoords="offset points",
                 ha='center')

plt.grid(True)
plt.show()

# -------------------------------
# SCATTER PLOT
# -------------------------------
hours = [1, 2, 3, 4, 5, 6, 7, 8]
scores = [40, 45, 50, 60, 65, 75, 80, 90]

plt.figure(figsize=(8,5))
plt.scatter(hours, scores,
            color='red',
            s=100)

plt.title("Hours Studied vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Score")

for i in range(len(hours)):
    plt.annotate(scores[i],
                 (hours[i], scores[i]),
                 xytext=(5,5),
                 textcoords='offset points')

plt.grid(True)
plt.show()

# -------------------------------
# HISTOGRAM
# -------------------------------
data = np.random.normal(70, 10, 200)

plt.figure(figsize=(8,5))
plt.hist(data,
         bins=10,
         color='orange',
         edgecolor='black')

plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")

plt.grid(axis='y', linestyle='--')
plt.show()