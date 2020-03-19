using Statistics
using CSV
using DataFrames
dataset = CSV.read("data/housingPriceData.csv")
id = dataset.id
price = dataset.price
bedrooms = dataset.bedrooms
bathrooms = dataset.bathrooms
sqft_living = dataset.sqft_living
m = length(id)
m_val = Int(round(m*60/100))
m_train = Int(round(m*80/100))

function stand(itr)
    d = sum(itr)/length(itr)
    e = std(itr)
    z = (itr .- d)/e
    return z
end

bedrooms = stand(bedrooms)
bathrooms = stand(bathrooms)
sqft_living = stand(sqft_living)
x0_train = ones(m_val)
bedrooms_train = bedrooms[1:m_val]
bathrooms_train = bathrooms[1:m_val]
sqft_living_train = sqft_living[1:m_val]
X_train = cat(x0_train, bedrooms_train, bathrooms_train, sqft_living_train, dims = 2)
Y_train = price[1:m_val]

x0_val = ones(m_train - m_val)
bedrooms_val = bedrooms[m_val+1:m_train]
bathrooms_val = bathrooms[m_val+1:m_train]
sqft_living_val = sqft_living[m_val+1:m_train]
X_val = cat(x0_val, bedrooms_val, bathrooms_val, sqft_living_val, dims = 2)
Y_val = price[m_val+1:m_train]

bedrooms_test = bedrooms[m_train+1:m]
bathrooms_test = bathrooms[m_train+1:m]
sqft_living_test = sqft_living[m_train+1:m]
x0_test = ones(m-m_train)
X_test = cat(x0_test, bedrooms_test, bathrooms_test, sqft_living_test, dims =2)
Y_test = price[m_train+1:m]

B = zeros(4, 1)

function costFunction(X, Y, B, rate)
    cost = sum(((X * B) - Y).^2) + (rate * sum((B).^2))
    return cost
end
rate = 0
initialcost = costFunction(X_train, Y_train, B, rate)

function gradientDescent(X, Y, B, rate, learningRate, numIterations)
    costHistory = zeros(numIterations)
    m = length(Y)
    for iteration in 1:numIterations
        loss = 2 * ((X * B) - Y)
        gradient = (X' * loss) + 2 * rate * B
        B = B - learningRate * gradient
        costHistory[iteration] = costFunction(X, Y, B, rate)
    end
    return B, costHistory
end

learningRate = 0.00001
rmse = zeros(100)
for i in 1:100
    rate = i+50
    newB, costHistory = gradientDescent(X_train, Y_train, B, rate, learningRate, 1000)
    YPred_val = X_val*newB
    rmse[i] = sqrt(sum((YPred_val - Y_val).^2)/(m_train - m_val))
end

function minimum_index(X)
    min = 1
    for iter = 1:length(X)
        if X[iter] < X[min]
            min = iter
        end
    end
    return min
end

alpha = 50 + minimum_index(rmse)

newB, costHistory = gradientDescent(X_train, Y_train, B, alpha, learningRate, 1000)
YPred_test = X_test*newB
rmse2 = sqrt(sum((YPred_test - Y_test).^2)/(m - m_train))
den = sum((Y_test .- mean(Y_test)).^2)
r2 = 1 - (sum((YPred_test - Y_test).^2)/den)

df = DataFrame(YPred_test)
CSV.write("data/2b.csv", df, writeheader = false)
