from abc import ABC, abstractmethod, abstractproperty


class FewShotMetric(ABC):

    @abstractmethod
    def score(self, samples, predictions):
        pass

    @abstractproperty
    def name(self):
        pass


class AccuracyMetric(FewShotMetric):

    def score(self, samples, predictions):
        sample2prediction = {prediction.sample: prediction.best_candidate.candidate for prediction in predictions}
        correct = total = 0
        for sample in samples:
            if sample.has_subproblems:
                correct += int(all([s.is_correct(sample2prediction[s]) for s in sample.subproblems]))
            else:
                correct += int(sample.is_correct(sample2prediction[sample]))
            total += 1
        return 100 * correct / total
    
    @property
    def name(self):
        return 'accuracy'


class SariMetric(FewShotMetric):

    def score(self, samples, predictions):
        sample2prediction = {prediction.sample: prediction.best_candidate.candidate for prediction in predictions}
        generations = []
        sources = []
        list_of_references = []
        for sample in samples:
            assert not sample.has_subproblems
            generations.append(sample2prediction[sample])
            sources.append(sample["source"])
            list_of_references.append(sample["references"])
        try:
            from easse.sari import corpus_sari
        except ImportError:
            raise ImportError("Please install EASSE from https://github.com/feralvam/easse/tree/master/easse")
        return corpus_sari(sources, generations, list(zip(*list_of_references)))
    
    @property
    def name(self):
        return 'sari'
