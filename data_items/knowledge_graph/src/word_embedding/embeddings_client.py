import selectors
import socket
import traceback
import json
import numpy as np

from . import libclient


def create_similarity_request(word1, word2):
    return dict(
        type="text/json",
        encoding="utf-8",
        content=dict(request='similarity', word1=word1, word2=word2),
    )


def creat_similarity_between_request(d: dict):
    return dict(
        type="text/json",
        encoding="utf-8",
        content=dict(request='column_names_similarity', dict=d),
    )


def create_embedding_request(word):
    return dict(
        type="text/json",
        encoding="utf-8",
        content=dict(request='embedding', word=word),
    )


def start_connection(host, port, sel, request):
    addr = (host, port)
    # print("starting connection to", addr)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)
    sock.connect_ex(addr)
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    message = libclient.Message(sel, sock, addr, request)
    sel.register(sock, events, data=message)


def n_similarity(mwe1, mwe2):
    mwe1 = ' '.join(mwe1)
    mwe2 = ' '.join(mwe2)

    sel = selectors.DefaultSelector()
    host, port = '127.0.0.1', 9600
    request = create_similarity_request(mwe1, mwe2)

    start_connection(host, port, sel, request)

    result = 0.0
    try:
        while True:
            events = sel.select(timeout=1)
            for key, mask in events:
                message = key.data
                try:
                    message.process_events(mask)
                except Exception:
                    print(
                        "main: error: exception for",
                        f"{message.addr}:\n{traceback.format_exc()}",
                    )
                    message.close()
            # Check for a socket being monitored to continue.
            if not sel.get_map():
                break
    except KeyboardInterrupt:
        print("caught keyboard interrupt, exiting")
    finally:
        sel.close()
    return message.response['result']


def get_embedding_of(word: str) -> np.ndarray:
    sel = selectors.DefaultSelector()
    host, port = '127.0.0.1', 9600
    request = create_embedding_request(word)

    start_connection(host, port, sel, request)

    result = []
    try:
        while True:
            events = sel.select(timeout=1)
            for key, mask in events:
                message = key.data
                try:
                    message.process_events(mask)
                except Exception:
                    print(
                        "main: error: exception for",
                        f"{message.addr}:\n{traceback.format_exc()}",
                    )
                    message.close()
            # Check for a socket being monitored to continue.
            if not sel.get_map():
                break
    except KeyboardInterrupt:
        print("caught keyboard interrupt, exiting")
    finally:
        sel.close()
    return np.array(message.response['result'])


def get_similarity_between(id_labels: dict):
    sel = selectors.DefaultSelector()
    host, port = '127.0.0.1', 9600
    request = creat_similarity_between_request(id_labels)

    start_connection(host, port, sel, request)

    result = 0.0
    try:
        while True:
            events = sel.select(timeout=1)
            for key, mask in events:
                message = key.data
                try:
                    message.process_events(mask)
                except Exception:
                    print(
                        "main: error: exception for",
                        f"{message.addr}:\n{traceback.format_exc()}",
                    )
                    message.close()
            # Check for a socket being monitored to continue.
            if not sel.get_map():
                break
    except KeyboardInterrupt:
        print("caught keyboard interrupt, exiting")
    finally:
        sel.close()
    return message.response['result']


if __name__ == '__main__':
    sim = n_similarity(['satellites'], ['moon'])
    print(sim)
