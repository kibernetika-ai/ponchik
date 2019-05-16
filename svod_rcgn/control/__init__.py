from svod_rcgn.control.listener import SVODListener, DEFAULT_LISTENER_PORT


def listener_args(args):
    return SVODListener(
        port=args.listener_port,
    )


def add_listener_args(parser):
    parser.add_argument(
        '--listener_port',
        type=int,
        default=DEFAULT_LISTENER_PORT,
        help='Listener port',
    )
