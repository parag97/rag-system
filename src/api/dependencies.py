from src.container.application_container import ApplicationContainer

_container = ApplicationContainer()


def get_container() -> ApplicationContainer:
    return _container