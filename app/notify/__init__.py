from app.notify import file, slack, print
from app.tools import utils

notifier = None
not_init_say = False
notifier_disabled = False


def init_notifier_args(args):
    if args.notify_slack_token and args.notify_slack_channel and args.notify_slack_server:
        init_notifier_slack(args.notify_slack_token, args.notify_slack_channel, args.notify_slack_server)
    elif args.notify_log:
        init_notifier_print()
    elif args.notify_file:
        init_notifier_file(args.notify_file)
    else:
        global notifier_disabled
        notifier_disabled = True


def init_notifier_slack(slack_token, slack_channel, slack_server):
    global notifier
    if slack_token and slack_channel and slack_server:
        notifier = slack.NotifySlack(slack_token, slack_channel, slack_server)
        if not notifier.is_ok():
            notifier = None


def init_notifier_print():
    global notifier
    notifier = print.NotifyPrint


def init_notifier_file(filename):
    global notifier
    notifier = file.NotifyFile(filename)


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
    parser.add_argument(
        '--notify_log',
        help='Enable notify to log.',
        action='store_true',
    )
    parser.add_argument(
        '--notify_file',
        help='Enable notify to file.',
        type=str,
        default=None,
    )


def notify(**kwargs):
    if notifier_disabled:
        return
    if notifier is None:
        global not_init_say
        if not not_init_say:
            utils.print_fun("=== Notifications aren't initialized ===")
            not_init_say = True
        return
    if not kwargs.get('name') and not kwargs.get('text'):
        kwargs['name'] = '--Somebody--'
    notifier.notify(**kwargs)
