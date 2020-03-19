
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
m_train = Int(round(m*80/100))
x0_train = ones(m_train)

function stand(itr)
    d = sum(itr)/length(itr)
    e = std(itr)
    z = (itr .- d)/e
    return z
end

bedrooms = stand(bedrooms)
bathrooms = stand(bathrooms)
sqft_living = stand(sqft_living)
bedrooms_train = bedrooms[1:m_train]
bathrooms_train = bathrooms[1:m_train]
sqft_living_train = sqft_living[1:m_train]
X_train = cat(x0_train, bedrooms_train, bathrooms_train, sqft_living_train, dims = 2)
Y_train = price[1:m_train]

bedrooms_test = bedrooms[m_train+1:m]
bathrooms_test = bathrooms[m_train+1:m]
sqft_living_test = sqft_living[m_train+1:m]
x0_test = ones(m-m_train)
X_test = cat(x0_test, bedrooms_test, bathrooms_test, sqft_living_test, dims =2)
Y_test = price[m_train+1:m]

B = zeros(4, 1)

function costFunction(X, Y, B)
    p = length(Y)
    cost = sum(((X * B) - Y).^2)/(2*p)
    return cost
end
intialCost = costFunction(X_train, Y_train, B);

function gradientDescent(X, Y, B, learningRate, numIterations)
    costHistory = zeros(numIterations)
    k = length(Y)
    for iteration in 1:numIterations
        loss = (X * B) - Y
        gradient = (X' * loss)/k
        B = B - learningRate * gradient
        cost = costFunction(X, Y, B)
        costHistory[iteration] = cost
    end
    return B, costHistory
end
rmse = zeros(75)
for i in 1:75
learningRate = i*0.0001
newB, costHistory = gradientDescent(X_train, Y_train, B, learningRate, 1000)
YPred_train = X_train*newB
rmse[i] = sqrt(sum((YPred_train - Y_train).^2)/(m_train))
end

learningRate = 0.005
newB, costHistory = gradientDescent(X_train, Y_train, B, learningRate, 1000)
YPred_test = X_test*newB
rmse2 = sqrt(sum((YPred_test - Y_test).^2)/(m - m_train))
den = sum((Y_test .- mean(Y_test)).^2)
r2 = 1 - (sum((YPred_test - Y_test).^2)/den)

df = DataFrame(YPred_test)
CSV.write("data/1a.csv", df, writeheader = false)
