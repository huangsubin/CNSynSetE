import numpy as np
from multiprocessing.dummy import Pool
import multiprocessing as multi
from types import MethodType, FunctionType
import warnings


def set_run_mode(func, mode, multiNum):
    '''

    :param func:
    :param mode: string
        can be  common, vectorization , parallel, cached
    :return:
    '''
    func.__dict__['mode'] = mode
    func.__dict__['multiNum'] = multiNum
    return


def func_transformer(func):
    '''
    transform this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x
        return x1 ** 2 + x2 ** 2 + x3 ** 2
    ```
    into this kind of function:
    ```
    def demo_func(x):
        x1, x2, x3 = x[:,0], x[:,1], x[:,2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    getting vectorial performance if possible:
    ```
    def demo_func(x):
        x1, x2, x3 = x[:, 0], x[:, 1], x[:, 2]
        return x1 ** 2 + (x2 - 0.05) ** 2 + x3 ** 2
    ```
    :param func:
    :return:
    '''

    # to support the former version
    if (func.__class__ is FunctionType) and (func.__code__.co_argcount > 1):
        warnings.warn('multi-input might be deprecated in the future, use fun(p) instead')

        def func_transformed(X):
            return np.array([func(*tuple(x)) for x in X])

        return func_transformed

    # to support the former version
    if (func.__class__ is MethodType) and (func.__code__.co_argcount > 2):
        warnings.warn('multi-input might be deprecated in the future, use fun(p) instead')

        def func_transformed(X):
            return np.array([func(tuple(x)) for x in X])

        return func_transformed

    # to support the former version
    if getattr(func, 'is_vector', False):
        warnings.warn('''
        func.is_vector will be deprecated in the future, use set_run_mode(func, 'vectorization') instead
        ''')
        set_run_mode(func, 'vectorization')

    mode = getattr(func, 'mode', 'others')  # vectorial, parallel, cached
    mNum = getattr(func, 'multiNum', 1)
    if mode == 'vectorization':
        return func
    # elif mode == 'cached':
    #     @lru_cache(maxsize=None)
    #     def func_cached(x):
    #         return func(x)
    #
    #     def func_warped(X):
    #         return np.array([func_cached(tuple(x)) for x in X])
    #
    #     return func_warped
    elif mode == 'parallel':

        # pool = Pool()
        pool = multi.Pool(processes=mNum)


        def func_transformed(X,optionsList):
            return np.array(pool.map(func, zip(X,range(len(X)),optionsList)))

        return func_transformed
    else:  # common
        def func_transformed(X,optionsList):
            return np.array([func(x) for x in zip(X,range(len(X)),optionsList)])

        return func_transformed

