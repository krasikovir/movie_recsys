from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

    
class KNNRecommender():

    '''
    Класс рекомендаций, основанный на методе KNN
    Методы:

    fit:
    Обучение модели. Генерация предполагаемых оценок пользователей, на основе которых будут выведены рекомендации.

    user_predict:
    Генерация 10 рекомендаций для пользователя среди тех объектов, которые он не видел. 
    '''
    
    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
        '''
        Метод принимает на вход датафрейм ratings, генерирует матрицу рейтингов user-item
        и заполняет пробелы на основе KNN.

        Также метод принимает на вход датафрейм movies, чтобы создать словарь названий фильмов по их Id

        Замечание: в данном датафрейме есть фильмы, которые никто не смотрел (их 3446, что есть треть от всех фильмов),
        и у них отсутсвуют рейтинги. Для данных фильмов я назначу одну случайню оценку от одного случайного юзера,
        чтобы обработать крайний случай. Т.к. KNN просто удаляет полностью nan'овый объект
        '''
        # original matrix
        self.om = ratings.pivot(index='userId',columns='movieId', values='rating')

        # names of empty movies
        empty_cols = self.om.loc[:, self.om.isna().sum() == self.om.shape[0] - 1].columns

        # filling one single rating randomly for each empty movie
        for col in empty_cols:
            self.om.loc[self.om.sample(n = 1).index, col] = np.random.randint(1,5)

        # restored matrix
        self.rm = KNNImputer(n_neighbors=30).fit_transform(self.om)
        self.rm = pd.DataFrame(self.rm, index = self.om.index, columns = self.om.columns)

        # dict with name by id
        self.name_by_id = pd.Series(movies.title.values,index=movies.movieId).to_dict()

    def user_predict(self, user_id: int) -> list:
        '''
        Метод принимает на вход id  user'а и возвращает 10 не просмотренных им фильмов в порядке убывания
        предполагаемой оценки. Возвращается лист из строк (названий фильмов)
        '''  

        if user_id not in self.om.index.values:
            return f"user_id = {user_id} doesn't exist"
        # unseen movies Ids for this user
        unseen_Ids = self.om[self.om.index == user_id].columns[self.om.isna().any()].tolist()

        # unseen movies with its rating prediction
        unseen_movies = self.rm[unseen_Ids][self.rm.index == user_id].reset_index(drop = True).T

        # top ten unseen movies' id
        top_id = unseen_movies.iloc[:,0].sort_values(ascending = False).index[:10].to_numpy()

        # get required names
        return [self.name_by_id[i] for i in top_id]