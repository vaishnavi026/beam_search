import matplotlib.pyplot as plt

# Given data
perplexity = [2.0455755377813194, 2.0007323301876867, 3.0829320756782335, 2.0440484945652977]
kl_divergence = [0.019396199504876945, 0.009905381842829636, 1.5635733925474735, 0.018002150602809783]
number_of_chars = [3000000, 2000000, 1000000, 4000000]

# Sorting based on number_of_chars for clearer visualization
sorted_indices = sorted(range(len(number_of_chars)), key=lambda k: number_of_chars[k])
number_of_chars = [number_of_chars[i] for i in sorted_indices]
perplexity = [perplexity[i] for i in sorted_indices]
kl_divergence = [kl_divergence[i] for i in sorted_indices]

# Plotting
plt.figure(figsize=(10,6))
plt.plot(number_of_chars, perplexity, label='Perplexity', marker='o')
plt.plot(number_of_chars, kl_divergence, label='KL Divergence', marker='x')
plt.xlabel('Number of Characters for Training')
plt.ylabel('Value')
plt.title('Perplexity and KL Divergence vs Number of Characters')
plt.legend()
plt.grid(True)
plt.show()

