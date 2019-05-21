from app.notify.slack import NotifySlack
from app.notify.print import NotifyPrint
from app.tools import print_fun

notifier = None
not_init_say = False


def init_notifier_params(slack_token, slack_channel, slack_server):
    global notifier
    if slack_token and slack_channel and slack_server:
        notifier = NotifySlack(slack_token, slack_channel, slack_server)
        if not notifier.is_ok():
            notifier = None
    if notifier is None:
        notifier = NotifyPrint


def init_notifier(args):
    init_notifier_params(args.notify_slack_token, args.notify_slack_channel, args.notify_slack_server)


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
    parser.add_argument(
        '--notify_slack_server',
        help='URL for slack server.',
        type=str,
        default=None,
    )


def notify(**kwargs):
    if notifier is None:
        global not_init_say
        if not_init_say:
            print_fun("=== Notifications aren't initialized ===")
            not_init_say = True
        return
    if not kwargs['name']:
        kwargs['name'] = '--Somebody--'
    notifier.notify(**kwargs)
