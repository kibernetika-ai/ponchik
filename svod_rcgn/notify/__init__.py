from svod_rcgn.notify.slack import NotifySlack
from svod_rcgn.notify.print import NotifyPrint
from svod_rcgn.tools.print import print_fun

notifier = None
not_init_say = False


def init_notifier(args):
    global notifier
    if args.notify_slack_token and args.notify_slack_channel:
        notifier = NotifySlack(args.notify_slack_token, args.notify_slack_channel)
    else:
        notifier = NotifyPrint()


def add_notify_args(parser):
    parser.add_argument(
        '--notify_slack_token',
        help='Slack token.',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--notify_slack_channel',
        help='Slack channel.',
        type=str,
        default=None,
    )


def notify(message):
    if notifier is None:
        global not_init_say
        if not_init_say:
            print_fun("=== Notifications aren't initialized ===")
            not_init_say = True
        return
    notifier.notify(message)
