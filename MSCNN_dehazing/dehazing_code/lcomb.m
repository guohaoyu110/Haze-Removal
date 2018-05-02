function A = lcomb(X, w, b)

    A = sum(bsxfun(@times, X, w), 3) + b;
    A = sigm(A);% 这就是推测出来的