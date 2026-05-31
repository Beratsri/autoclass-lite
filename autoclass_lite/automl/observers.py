from abc import ABC, abstractmethod


class Observer(ABC):
    """
    Abstract observer. Subclasses receive event notifications from AutoML (Observer pattern).
    """
    @abstractmethod
    def update(self, event:str, data: dict) -> None:
        """
        Called when the subject fires an event.
        """
        pass

class ConsoleObserver(Observer):
    """
    Prints training progress to the console.
    """
    def update(self, event: str, data: dict) -> None:
        """Print a progress message for fit_start, model_done, and fit_done events."""
        if event == "model_done":
            print(f"  [{data['model']}] {data['scores']}")
        elif event == "fit_start":
            print("Starting AutoML...")
        elif event == "fit_done":
            print(f"Done. Best model: {data['best_model']}")


    